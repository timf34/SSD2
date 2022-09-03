def print_model_params(_model) -> None:
    _sum = sum(p.numel() for p in _model.parameters())
    print(f"total parameters {_sum}")