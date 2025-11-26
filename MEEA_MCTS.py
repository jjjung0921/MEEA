"""
MCTS-leaning version of MEEA.

This variant keeps the original MCTS-style tree policy (PUCT) and adds optional
Dirichlet noise at the root to encourage exploration, removing the A*-like
best-of-openings selection used in MEEA*.
"""

import argparse
import numpy as np
import torch

from MEEA_PC_parallel import (
    MinMaxStats,
    Node,
    prepare_expand,
    prepare_starting_molecules,
    prepare_value,
    value_fn,
)


class MEEAMCTS:
    """
    Pure PUCT rollout without the best-first tie-in.
    """

    def __init__(
        self,
        target_mol,
        known_mols,
        value_model,
        expand_fn,
        device,
        cpuct=4.0,
        topk=50,
        max_rollouts=800,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        thread_id=None,
    ):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.cpuct = cpuct
        self.thread_id = thread_id
        self.topk = topk
        self.max_rollouts = max_rollouts
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

        root_value = value_fn(self.value_model, [target_mol], self.device)
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=self.cpuct)

        self.visited_policy = {}
        self.visited_state = []
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)
        self.iterations = 0  # counts policy model invocations
        self.last_logged_iteration = -1

    def _select_leaf(self):
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            best_move = current.select_child(self.min_max_stats)
            current = current.children[best_move]

    def _add_dirichlet_noise(self, priors):
        if self.dirichlet_alpha is None or self.dirichlet_frac is None:
            return priors
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
        return (1 - self.dirichlet_frac) * priors + self.dirichlet_frac * noise

    def expand(self, node):
        node.is_expanded = True
        expanded_mol = node.state[0]

        if expanded_mol in self.visited_policy:
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.expand_fn.run(expanded_mol, topk=self.topk)
            self.iterations += 1
            self.visited_policy[expanded_mol] = expanded_policy.copy() if (
                expanded_policy is not None and len(expanded_policy["scores"]) > 0
            ) else None

        if expanded_policy is not None and len(expanded_policy["scores"]) > 0:
            node.child_illegal = np.array([0] * len(expanded_policy["scores"]))

            priors = np.array([1.0 / len(expanded_policy["scores"])] * len(expanded_policy["scores"]))
            if node.parent is None:
                priors = self._add_dirichlet_noise(priors)

            for i in range(len(expanded_policy["scores"])):
                reactant = [r for r in expanded_policy["reactants"][i].split(".") if r not in self.known_mols]
                reactant = reactant + node.state[1:]
                reactant = sorted(list(set(reactant)))

                cost = -np.log(np.clip(expanded_policy["scores"][i], 1e-3, 1.0))
                template = expanded_policy["template"][i]
                reaction = expanded_policy["reactants"][i] + ">>" + expanded_mol

                if len(reactant) == 0:
                    child = Node(
                        [],
                        0,
                        cost=cost,
                        prior=priors[i],
                        action_mol=expanded_mol,
                        reaction=reaction,
                        fmove=len(node.children),
                        template=template,
                        parent=node,
                        cpuct=self.cpuct,
                    )
                    return True, child
                else:
                    h = value_fn(self.value_model, reactant, self.device)
                    child = Node(
                        reactant,
                        h,
                        cost=cost,
                        prior=priors[i],
                        action_mol=expanded_mol,
                        reaction=reaction,
                        fmove=len(node.children),
                        template=template,
                        parent=node,
                        cpuct=self.cpuct,
                    )

                    if ".".join(reactant) in self.visited_state:
                        node.child_illegal[child.fmove] = 1000
                        back_check_node = node
                        while back_check_node.parent is not None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                            back_check_node = back_check_node.parent
        else:
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1000
                back_check_node = node.parent
                while (
                    back_check_node is not None
                    and back_check_node.parent is not None
                    and np.all(back_check_node.child_illegal > 0)
                ):
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

    def search(self):
        success, node = False, None
        progress_interval = 100

        while (
            self.iterations < self.max_rollouts
            and not success
            and (not np.all(self.root.child_illegal > 0) or len(self.root.child_illegal) == 0)
        ):
            expand_node = self._select_leaf()

            if ".".join(expand_node.state) in self.visited_state:
                expand_node.parent.child_illegal[expand_node.fmove] = 1000
                back_check_node = expand_node.parent
                while back_check_node is not None and back_check_node.parent is not None and np.all(
                    back_check_node.child_illegal > 0
                ):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
                continue
            else:
                self.visited_state.append(".".join(expand_node.state))
                success, node = self.expand(expand_node)
                self.update(expand_node)

            if (
                self.iterations > self.last_logged_iteration
                and (self.iterations % progress_interval == 0 or self.iterations == 1)
                and self.iterations <= self.max_rollouts
            ):
                prefix = f"[Thread {self.thread_id}] " if self.thread_id is not None else ""
                print(f"{prefix}[MCTS] target={self.target_mol} iterations={self.iterations}/{self.max_rollouts}", flush=True)
                self.last_logged_iteration = self.iterations

            if self.visited_policy.get(self.target_mol) is None:
                return False, None, self.max_rollouts

        return success, node, self.iterations

    @staticmethod
    def vis_synthetic_path(node):
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


def _default_device(gpu_id):
    return torch.device("cpu") if gpu_id is None or gpu_id < 0 else torch.device(f"cuda:{gpu_id}")


def main():
    parser = argparse.ArgumentParser(description="MCTS-leaning MEEA search for a single target molecule.")
    parser.add_argument("--target", required=True, help="Target molecule in SMILES format.")
    parser.add_argument("--policy", default="./saved_model/policy_model.ckpt", help="Path to policy model checkpoint.")
    parser.add_argument("--value", default="./saved_model/value_pc.pt", help="Path to value model checkpoint.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use; -1 for CPU.")
    parser.add_argument("--topk", type=int, default=50, help="Number of reactions to keep from policy model.")
    parser.add_argument("--cpuct", type=float, default=4.0, help="Exploration constant for PUCT.")
    parser.add_argument("--max-rollouts", type=int, default=800, help="Number of policy calls / rollouts to run.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25, help="Weight of Dirichlet noise at the root.")
    args = parser.parse_args()

    known_mols = prepare_starting_molecules()
    device = _default_device(args.gpu)
    expand_fn = prepare_expand(args.policy, args.gpu)
    value_model = prepare_value(args.value, args.gpu)

    planner = MEEAMCTS(
        args.target,
        known_mols,
        value_model,
        expand_fn,
        device,
        cpuct=args.cpuct,
        topk=args.topk,
        max_rollouts=args.max_rollouts,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
    )
    success, node, calls = planner.search()
    route, templates = planner.vis_synthetic_path(node)

    # Save results so multiple runs can be collected in one place.
    with open("MCTS.txt", "a") as f:
        f.write(
            f"[MCTS] Target: {args.target}\n"
            f"Success: {success}\n"
            f"Iterations (policy calls): {calls}\n"
            f"Route: {route}\n"
            f"Templates: {templates}\n"
            "---\n"
        )

    print(f"Success: {success}")
    print(f"Iterations (policy calls): {calls}")
    print("Route:", route)
    print("Templates:", templates)


if __name__ == "__main__":
    main()
