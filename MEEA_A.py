"""
A*-leaning version of MEEA.

This script keeps the same expansion (policy / value) models that MEEA uses,
but drives the search with a classic A* frontier: always expand the node with
the lowest f = g + h and do not add PUCT-style exploration bonuses. The goal
is to provide a deterministic, greedy best-first baseline alongside the
exploration-heavy MCTS variant.
"""

import argparse
import heapq
import numpy as np
import torch

from MEEA_PC_parallel import (
    Node,
    prepare_expand,
    prepare_starting_molecules,
    prepare_value,
    value_fn,
)


class MEEAAStar:
    """
    A*-style search that reuses the MEEA models but disables PUCT exploration.
    """

    def __init__(
        self,
        target_mol,
        known_mols,
        value_model,
        expand_fn,
        device,
        topk=50,
        max_expansions=500,
    ):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_model = value_model
        self.expand_fn = expand_fn
        self.device = device
        self.topk = topk
        self.max_expansions = max_expansions

        root_value = value_fn(self.value_model, [target_mol], self.device)
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=0.0)

        self.frontier = []
        self.counter = 0  # tie-breaker for heap
        heapq.heappush(self.frontier, (self.root.f, self.counter, self.root))

        self.visited_policy = {}
        self.visited_state = set()
        self.iterations = 0  # counts policy model invocations

    def _expand_node(self, node):
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

        if expanded_policy is None or len(expanded_policy["scores"]) == 0:
            return []

        children = []
        priors = np.array([1.0 / len(expanded_policy["scores"])] * len(expanded_policy["scores"]))

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
                    cpuct=0.0,
                )
                children.append(child)
            else:
                state_key = ".".join(reactant)
                if state_key in self.visited_state:
                    continue
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
                    cpuct=0.0,
                )
                children.append(child)

        return children

    def search(self):
        """
        Run best-first search up to max_expansions.
        """
        while self.frontier and self.iterations < self.max_expansions:
            _, _, node = heapq.heappop(self.frontier)

            if len(node.state) == 0:
                return True, node, self.iterations

            state_key = ".".join(node.state)
            if state_key in self.visited_state:
                continue
            self.visited_state.add(state_key)

            for child in self._expand_node(node):
                if len(child.state) == 0:
                    return True, child, self.iterations
                self.counter += 1
                heapq.heappush(self.frontier, (child.f, self.counter, child))

        return False, None, self.iterations

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
    parser = argparse.ArgumentParser(description="A*-leaning MEEA search for a single target molecule.")
    parser.add_argument("--target", required=True, help="Target molecule in SMILES format.")
    parser.add_argument("--policy", default="./saved_model/policy_model.ckpt", help="Path to policy model checkpoint.")
    parser.add_argument("--value", default="./saved_model/value_pc.pt", help="Path to value model checkpoint.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use; -1 for CPU.")
    parser.add_argument("--topk", type=int, default=50, help="Number of reactions to keep from policy model.")
    parser.add_argument("--max-expansions", type=int, default=500, help="Maximum policy calls / expansions.")
    args = parser.parse_args()

    known_mols = prepare_starting_molecules()
    device = _default_device(args.gpu)
    expand_fn = prepare_expand(args.policy, args.gpu)
    value_model = prepare_value(args.value, args.gpu)

    planner = MEEAAStar(
        args.target,
        known_mols,
        value_model,
        expand_fn,
        device,
        topk=args.topk,
        max_expansions=args.max_expansions,
    )
    success, node, calls = planner.search()
    route, templates = planner.vis_synthetic_path(node)

    # Save results so multiple runs can be collected in one place.
    with open("A_star.txt", "a") as f:
        f.write(
            f"Target: {args.target}\n"
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
