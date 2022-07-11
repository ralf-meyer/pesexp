from pesexp.cli.params import parse_opt_args


def test_opt_defaults():
    params = parse_opt_args(["terachem.inp"])
    assert params["coords"] == "dlc"
    assert params["maxiter"] == 300
    assert params["hessian_guess"] == "trivial"
    assert params["hessian_thresh"] is None


def test_parse_opt_args():
    params = parse_opt_args(
        [
            "--coords",
            "cart",
            "--maxiter",
            "400",
            "--hessian_guess",
            "lindh",
            "--hessian_thresh",
            "1e-3",
            "terachem.inp",
        ]
    )
    assert params["coords"] == "cart"
    assert params["maxiter"] == 400
    assert params["hessian_guess"] == "lindh"
    assert params["hessian_thresh"] == 0.001
