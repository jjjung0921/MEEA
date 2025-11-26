import argparse

from policyNet_variants import train_policy_variant


def parse_args():
    parser = argparse.ArgumentParser(description="Train dropout-MLP policy network")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout-rate", type=float, default=0.4)
    parser.add_argument("--saved-model", default="./model/dropout_policy")
    parser.add_argument("--fp-dim", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    train_policy_variant(
        model_type="dropout",
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        saved_model=args.saved_model,
        fp_dim=args.fp_dim,
        hidden=args.hidden,
    )


if __name__ == "__main__":
    main()
