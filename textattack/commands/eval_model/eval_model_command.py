from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import scipy
import torch

import textattack
from textattack.commands import TextAttackCommand
from textattack.commands.attack.attack_args import (
    HUGGINGFACE_DATASET_BY_MODEL,
    TEXTATTACK_DATASET_BY_MODEL,
)
from textattack.commands.attack.attack_args_helpers import (
    add_dataset_args,
    add_model_args,
    parse_dataset_from_args,
    parse_model_from_args,
)

logger = textattack.shared.logger


def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


class EvalModelCommand(TextAttackCommand):
    """The TextAttack model benchmarking module:

    A command line parser to evaluatate a model from user
    specifications.
    """

    def get_preds(self, model, inputs, batch_size):
        with torch.no_grad():
            preds = textattack.shared.utils.batch_model_predict(model, inputs, batch_size)
        return preds

    def test_model_on_dataset(self, args):
        model = parse_model_from_args(args)
        dataset = parse_dataset_from_args(args)

        dataset = dataset[:args.num_examples]
        attacked_texts = [textattack.shared.AttackedText(text) for text, _ in dataset]
        ground_truth_outputs = [label for _, label in dataset]

        inputs = textattack.shared.utils.batch_tokenize(model.tokenizer, attacked_texts, batch_size=args.batch_size)
        preds = self.get_preds(model, inputs, batch_size=args.batch_size)

        preds = preds.cpu()
        ground_truth_outputs = torch.tensor(ground_truth_outputs).cpu()

        logger.info(f"Got {len(preds)} predictions.")

        if preds.ndim == 1:
            # if preds is just a list of numbers, assume regression for now
            # TODO integrate with `textattack.metrics` package
            pearson_correlation, _ = scipy.stats.pearsonr(ground_truth_outputs, preds)
            spearman_correlation, _ = scipy.stats.spearmanr(ground_truth_outputs, preds)

            logger.info(f"Pearson correlation = {_cb(pearson_correlation)}")
            logger.info(f"Spearman correlation = {_cb(spearman_correlation)}")
        else:
            guess_labels = preds.argmax(dim=1)
            successes = (guess_labels == ground_truth_outputs).sum().item()
            perc_accuracy = successes / len(preds) * 100.0
            perc_accuracy = "{:.2f}%".format(perc_accuracy)
            logger.info(f"Successes {successes}/{len(preds)} ({_cb(perc_accuracy)})")

    def run(self, args):
        # Default to 'all' if no model chosen.
        if not (args.model or args.model_from_huggingface or args.model_from_file):
            for model_name in list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
                TEXTATTACK_DATASET_BY_MODEL.keys()
            ):
                args.model = model_name
                self.test_model_on_dataset(args)
                logger.info("-" * 50)
        else:
            self.test_model_on_dataset(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "eval",
            help="evaluate a model with TextAttack",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

        add_model_args(parser)
        add_dataset_args(parser)

        parser.add_argument(
            "--batch-size",
            type=int,
            default=256,
            help="Batch size for model inference.",
        )
        parser.set_defaults(func=EvalModelCommand())
