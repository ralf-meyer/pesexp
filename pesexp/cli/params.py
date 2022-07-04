import argparse


def parse_opt_args(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Terachem input file")
    preproc = parser.add_argument_group(
        "preprocessing", "Options for preprocessing steps"
    )
    preproc.add_argument("--preopt", type=str, help="Currently unused")
    opt = parser.add_argument_group(
        "optimization", "Options for the main geometry " "optimization procedure"
    )
    opt.add_argument("--optimizer", type=str, choices=["BFGS", "RFO"])
    opt.add_argument(
        "--coords", type=str, choices=["cart", "dlc", "anc"], default="dlc"
    )
    opt.add_argument(
        "--maxiter",
        type=int,
        default=300,
        help="Maximum number of iterations, defaults to 300",
    )
    opt.add_argument(
        "--hessian_guess",
        type=str,
        default="trivial",
        choices=["trivial", "schlegel", "fischer_almloef", "lindh"],
        help="Method to generate the initial hessian guess",
    )
    opt.add_argument("--hessian_thresh", type=float, help="")

    optimize_args = parser.parse_args(args)
    # Convert to dict following
    # https://docs.python.org/3/library/argparse.html#the-namespace-object
    args_dict = {}
    for key, value in vars(optimize_args).items():
        args_dict[key] = value

    return args_dict
