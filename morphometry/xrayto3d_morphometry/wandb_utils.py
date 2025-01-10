import wandb


def get_run_from_model_name(model_name, wandb_runs):
    """return the first wandb run with given model name"""
    for run in wandb_runs:
        if model_name == run.config["MODEL_NAME"]:
            return run
    raise ValueError(f"{model_name} not found")


def filter_wandb_run(
    anatomy: str,
    project_name="msrepo/2d-3d-benchmark",
    tags=("model-compare",),
    verbose=False,
):
    """find wandb runs that fulfil given criteria"""
    api = wandb.Api()
    filters_mongodb_query_operation = {}
    if len(tags) <= 1:
        filters_mongodb_query_operation["tags"] = {"$in": tags}
    else:
        filters_mongodb_query_operation["$and"] = [{"tags": {"$in": [k]}} for k in tags]
    runs = api.runs(project_name, filters=filters_mongodb_query_operation)
    if verbose:
        print(f"found {len(runs)} unfiltered runs")

    filtered_runs = [
        run
        for run in runs
        if "ANATOMY" in run.config and anatomy in run.config["ANATOMY"]
    ]

    if verbose:
        print(f"got {len(filtered_runs)} runs after filtering anatomy={anatomy}")

    return filtered_runs
