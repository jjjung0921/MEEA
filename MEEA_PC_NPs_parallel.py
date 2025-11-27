import argparse
import pickle
from multiprocessing import Process
import multiprocessing
import torch
import numpy as np
from valueEnsemble import ValueEnsemble
import signal
import os
import time
from contextlib import contextmanager
from policyNet import MLPModel
from policyNet_variants import PolicyVariantModel
import warnings
warnings.filterwarnings('ignore')  # RDKit 및 기타 deprecation 경고 억제
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # RDKit 로거 비활성화
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_dim)
    fp = generator.GetFingerprint(mol)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr


def batch_smiles_to_fp(s_list, fp_dim=2048):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps


class MinMaxStats(object):
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value


def prepare_starting_molecules_natural():
    fr = open('./prepare_data/building_blocks.txt', 'r')
    data = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        data.append(line[-1])
    return data


def prepare_expand(
    model_path,
    template_path,
    gpu=-1,
    policy_type='mlp',
    fp_dim=2048,
    transformer_kwargs=None,
    dropout_rate=None,
):
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    if policy_type == 'mlp':
        return MLPModel(model_path, template_path, device=device, fp_dim=fp_dim)
    elif policy_type == 'dropout':
        return MLPModel(model_path, template_path, device=device, fp_dim=fp_dim)
    elif policy_type == 'transformer':
        kwargs = transformer_kwargs or {}
        return PolicyVariantModel(
            model_type='transformer',
            state_path=model_path,
            template_path=template_path,
            device=device,
            fp_dim=fp_dim,
            model_kwargs=kwargs
        )
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")


def prepare_value(model_f, gpu=None):
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model


def value_fn(model, mols, device):
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols, fp_dim=2048).reshape(num_mols, -1)
    index = len(fps)
    if len(fps) <= 5:
        mask = np.ones(5)
        mask[index:] = 0
        fps_input = np.zeros((5, 2048))
        fps_input[:index, :] = fps
    else:
        mask = np.ones(len(fps))
        fps_input = fps
    fps = torch.FloatTensor([fps_input.astype(np.float32)]).to(device)
    mask = torch.FloatTensor([mask.astype(np.float32)]).to(device)
    v = model(fps, mask).cpu().data.numpy()
    return v[0][0]

class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None, template=None, parent=None, cpuct=1.5):
        self.state = state
        self.cost = cost
        self.h = h
        self.prior = prior
        self.visited_time = 0
        self.is_expanded = False
        self.template = template
        self.action_mol = action_mol
        self.fmove = fmove
        self.reaction = reaction
        self.parent = parent
        self.cpuct = cpuct
        self.children = []
        self.child_illegal = np.array([])
        if parent is not None:
            self.g = self.parent.g + cost
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.g = 0
            self.depth = 0
        self.f = self.g + self.h
        self.f_mean_path = []

    def child_N(self):
        N = [child.visited_time for child in self.children]
        return np.array(N)

    def child_p(self):
        prior = [child.prior for child in self.children]
        return np.array(prior)

    def child_U(self):
        child_Ns = self.child_N() + 1
        prior = self.child_p()
        child_Us = self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns
        return child_Us

    def child_Q(self, min_max_stats):
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, min_max_stats):
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal
        best_move = np.argmax(action_score)
        return best_move


class MCTS_A:
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device, simulations, cpuct, thread_id=None):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.cpuct = cpuct
        self.thread_id = thread_id  # Thread 번호 저장
        root_value = value_fn(self.value_model, [target_mol], self.device)
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=self.cpuct)
        self.open = [self.root]
        self.visited_policy = {}
        self.visited_state = []
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)
        self.opening_size = simulations
        self.iterations = 0
        self.last_logged_iteration = -1

    def select_a_leaf(self):
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            best_move = current.select_child(self.min_max_stats)
            current = current.children[best_move]

    def select(self):
        openings = [self.select_a_leaf() for _ in range(self.opening_size)]
        stats = [opening.f for opening in openings]
        index = np.argmin(stats)
        return openings[index]
    
    def preprocessPolicy(self, policy):
        # policy가 None인 경우 처리
        if policy is None:
            return None

        scores = []
        reactants = []
        templates = []
        for i in range(len(policy['scores'])):
            if len(policy['reactants'][i]) != 0:
                scores.append(policy['scores'][i])
                reactants.append(policy['reactants'][i])
                templates.append(policy['template'][i])
        ans_policy = {
            'scores': scores,
            'reactants': reactants,
            'template': templates
        }
        return ans_policy

    def expand(self, node):
        node.is_expanded = True
        expanded_mol_index = 0
        expanded_mol = node.state[expanded_mol_index]
        if expanded_mol in self.visited_policy.keys():
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.preprocessPolicy(self.expand_fn.run(expanded_mol))
            self.iterations += 1
            if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
                self.visited_policy[expanded_mol] = expanded_policy.copy()
            else:
                self.visited_policy[expanded_mol] = None
        if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
            node.child_illegal = np.array([0] * len(expanded_policy['scores']))
            for i in range(len(expanded_policy['scores'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                reactant = reactant + node.state[: expanded_mol_index] + node.state[expanded_mol_index + 1:]
                reactant = sorted(list(set(reactant)))
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol
                priors = np.array([1.0 / len(expanded_policy['scores'])] * len(expanded_policy['scores']))
                if len(reactant) == 0:
                    child = Node([], 0, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    return True, child
                else:
                    h = value_fn(self.value_model, reactant, self.device)
                    child = Node(reactant, h, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    if '.'.join(reactant) in self.visited_state:
                        node.child_illegal[child.fmove] = 1000
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                            back_check_node = back_check_node.parent
        else:
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1000
                back_check_node = node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
        return False, None

    def update(self, node):
        stat = node.f
        self.min_max_stats.update(stat)
        current = node
        while current is not None:
            current.f_mean_path.append(stat)
            current = current.parent

    def search(self, times=500):
        success, node = False, None
        progress_interval = 100  # 100회마다 진행 상황 출력

        while self.iterations < times and not success and (not np.all(self.root.child_illegal > 0) or len(self.root.child_illegal) == 0):
            expand_node = self.select()
            if '.'.join(expand_node.state) in self.visited_state:
                expand_node.parent.child_illegal[expand_node.fmove] = 1000
                back_check_node = expand_node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
                continue
            else:
                self.visited_state.append('.'.join(expand_node.state))
                success, node = self.expand(expand_node)
                self.update(expand_node)

            # 진행 상황 출력
            if (
                self.iterations > self.last_logged_iteration
                and (self.iterations % progress_interval == 0 or self.iterations == 1)
                and self.iterations <= times
            ):
                thread_prefix = f"[Thread {self.thread_id}] " if self.thread_id is not None else ""
                print(f"{thread_prefix}[MCTS] target={self.target_mol} iterations={self.iterations}/{times}", flush=True)
                self.last_logged_iteration = self.iterations

            if self.visited_policy[self.target_mol] is None:
                return False, None, times
        return success, node, self.iterations

    def vis_synthetic_path(self, node):
        if node is None:
            return [], []
        reaction_path = []
        template_path = []
        current = node
        while current is not None:
            reaction_path.append(current.reaction)
            template_path.append(current.template)
            current = current.parent
        return reaction_path[::-1], template_path[::-1]


def play(dataset, mols, thread, known_mols, value_model, expand_fn, device, simulations, cpuct, times=500):
    # 멀티프로세싱 환경: 각 자식 프로세스에서 RDKit 로거 비활성화
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    routes = []
    templates = []
    successes = []
    depths = []
    counts = []
    total = len(mols)

    for idx, mol in enumerate(mols, 1):
        print(f"[Thread {thread}] Start molecule {idx}/{total}: {mol}", flush=True)
        try:
            with time_limit(600):
                player = MCTS_A(mol, known_mols, value_model, expand_fn, device, simulations, cpuct, thread_id=thread)
                success, node, count = player.search(times=times)
                route, template = player.vis_synthetic_path(node)
        except Exception as e:
            success = False
            route = [None]
            template = [None]
            print(f"[Thread {thread}] Molecule {idx}/{total} failed with error: {type(e).__name__}: {str(e)}", flush=True)

        routes.append(route)
        templates.append(template)
        successes.append(success)
        if success:
            depths.append(node.depth)
            counts.append(count)
            print(f"[Thread {thread}] Completed molecule {idx}/{total} in {count} iterations (depth {node.depth})", flush=True)
        else:
            depths.append(32)
            counts.append(-1)
            print(f"[Thread {thread}] Failed molecule {idx}/{total}", flush=True)
    ans = {
        'route': routes,
        'template': templates,
        'success': successes,
        'depth': depths,
        'counts': counts
    }
    with open('./test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(thread) + '.pkl', 'wb') as writer:
        pickle.dump(ans, writer, protocol=4)


def gather(dataset, simulations, cpuct, times, elapsed_time, policy_name, np_mode=True):
    result = {
        'route': [],
        'template': [],
        'success': [],
        'depth': [],
        'counts': []
    }
    processed = 0
    for i in range(28):
        file = './test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(i) + '.pkl'
        if not os.path.exists(file):
            continue
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for key in result.keys():
            result[key] += data[key]
        processed += 1
        os.remove(file)

    if processed == 0:
        print("No worker results found; skipping aggregation.", flush=True)
        return
    success = np.mean(result['success'])
    depth = np.mean(result['depth'])
    fr = open('result_simulation.txt', 'a')
    # Format: dataset, simulations, times, cpuct, success_rate, avg_depth, elapsed_time(s), elapsed_time(h:m:s)
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    fr.write(f"{dataset}\t{simulations}\t{times}\t{cpuct}\t{success:.4f}\t{depth:.2f}\t{elapsed_time:.2f}\t{time_str}\t{policy_name}\t{'NP' if np_mode else 'PC'}\n")
    f = open('./test/stat_norm_retro_' + dataset + '_' + str(simulations) + '_' + str(cpuct) + '_' + str(times) + '.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MEEA PC NP planner')
    parser.add_argument('--policy-type', default='mlp', choices=['mlp', 'dropout', 'transformer'], help='policy network type')
    parser.add_argument('--policy-model', default='./saved_model/policy_model.ckpt', help='path to policy model checkpoint')
    parser.add_argument('--template-path', default='./saved_model/template_rules.dat', help='path to template rules file')
    parser.add_argument('--fp-dim', type=int, default=2048, help='fingerprint dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.4, help='dropout rate for dropout policy')
    # Transformer-specific
    parser.add_argument('--patch-size', type=int, default=32, help='transformer patch size')
    parser.add_argument('--d-model', type=int, default=128, help='transformer embedding dim')
    parser.add_argument('--nhead', type=int, default=4, help='transformer num heads')
    parser.add_argument('--num-layers', type=int, default=2, help='transformer encoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=256, help='transformer ff dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='transformer dropout')
    args = parser.parse_args()

    transformer_kwargs = dict(
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    known_mols = prepare_starting_molecules_natural()
    simulations = 100
    cpuct = 4.0
    multiprocessing.set_start_method('spawn', force=True)
    one_steps = []
    devices = []
    value_models = []
    model_path = args.policy_model
    model_f = './saved_model/value_pc.pt'
    gpus = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(gpus)):
        one_step = prepare_expand(
            model_path,
            args.template_path,
            gpus[i],
            policy_type=args.policy_type,
            fp_dim=args.fp_dim,
            transformer_kwargs=transformer_kwargs if args.policy_type == 'transformer' else None,
            dropout_rate=args.dropout_rate if args.policy_type == 'dropout' else None
        )
        device = torch.device('cuda:' + str(gpus[i]))
        value_model = prepare_value(model_f, gpus[i])
        value_models.append(value_model)
        one_steps.append(one_step)
        devices.append(device)
    for dataset in ['natural']:
        fileName = './test_dataset/' + dataset + '.pkl'
        with open(fileName, 'rb') as f:
            targets = pickle.load(f)
        intervals = int(len(targets) / len(gpus))
        num_more = len(targets) - intervals * len(gpus)
        for simulations in [100]:
            for times in [3000]:
                start_time = time.time()
                jobs = [Process(target=play, args=(dataset, targets[i * (intervals + 1): (i + 1) * (intervals + 1)], i, known_mols, value_models[i], one_steps[i], devices[i], simulations, cpuct, times)) for i in range(num_more)]
                start = num_more * (intervals + 1)
                for i in range(len(gpus)- num_more):
                    jobs.append(Process(target=play, args=(dataset, targets[start + i * intervals: start + (i + 1) * intervals], num_more + i, known_mols, value_models[num_more + i], one_steps[num_more + i], devices[num_more + i], simulations, cpuct, times)))
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                elapsed_time = time.time() - start_time
                policy_name = os.path.basename(model_path)
                gather(dataset, simulations, cpuct, times, elapsed_time, policy_name, np_mode=True)
