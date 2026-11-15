"""Independent NumPy patch extraction and direct-solve replay of final revision heads."""

import json
from pathlib import Path
import numpy as np
import yaml
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from patchloss.selection.revision_data import load_revision
from patchloss.selection.revision import write_json


def independent_features(images, p, W):
    n, c, h, w = images.shape
    patches = images.reshape(n, c, h//p, p, w//p, p).transpose(0, 2, 4, 1, 3, 5)
    return (patches.reshape(n, -1, c*p*p) @ W.T).reshape(n, -1)


def verify():
    config = yaml.safe_load(Path("configs/selection_revision.yaml").read_text())
    root = Path(config["output"])
    checked, largest = 0, 0.
    records = []
    for domain in config["datasets"]:
        for seed in config["seeds"]:
            data, md = load_revision(domain, seed, config)
            training = np.concatenate([data[k][0] for k in ["operator", "diagnostic"]])
            labels = np.concatenate([data[k][1] for k in ["operator", "diagnostic"]])
            for budget in config["budgets"]:
                path = root/f"{domain}-b{budget}-s{seed}"
                expected_md = json.loads((path/"datasets.json").read_text())
                assert md == expected_md, "Reconstructed official splits differ"
                pretest = json.loads((path/"pretest.json").read_text())
                predictions = np.load(path/"predictions.npz")
                for row in pretest["rows"]:
                    cid = row["candidate"]
                    W = np.load(path/f"operator_{cid}.npz")["matrix"]
                    saved = np.load(path/f"head_{cid}.npz")
                    features = independent_features(training, row["patch_size"], W)
                    mean = features.mean(0)
                    scale = np.sqrt(np.mean((features-mean)**2))
                    x = (features-mean)/scale
                    targets = (labels[:,None] == saved["classes"]).astype(float)
                    prior = targets.mean(0)
                    coef = cho_solve(cho_factor(x.T@x + float(saved["alpha"])*np.eye(budget)), x.T@(targets-prior))
                    error = float(np.max(np.abs(coef-saved["coef"])))
                    largest = max(largest,error)
                    np.testing.assert_allclose(coef, saved["coef"],rtol=1e-6,atol=1e-9)
                    np.testing.assert_allclose(mean,saved["mean"],rtol=1e-13,atol=1e-13)
                    np.testing.assert_allclose(scale,saved["scale"],rtol=1e-13)
                    val_x = independent_features(data["validation"][0], row["patch_size"], W)
                    vp = saved["classes"][(((val_x-mean)/scale)@coef+prior).argmax(1)]
                    score = np.mean(vp == data["validation"][1])
                    curve = [r for r in pretest["validation_curves"] if r["candidate"]==cid]
                    chosen = max(curve,key=lambda r:(r["validation_accuracy"],r["alpha"]))
                    assert chosen["alpha"] == float(saved["alpha"])
                    np.testing.assert_allclose(score,chosen["validation_accuracy"],atol=1e-15)
                    test = data["test"][0]
                    predicted = []
                    for i in range(0,len(test),512):
                        z = independent_features(test[i:i+512], row["patch_size"], W)
                        # Replay stored coefficients to test extraction/prediction independently.
                        logits = ((z-saved["mean"])/saved["scale"])@saved["coef"]+saved["prior"]
                        predicted.append(saved["classes"][logits.argmax(1)])
                    np.testing.assert_array_equal(np.concatenate(predicted),predictions[cid])
                    checked += 1
                    records.append(dict(dataset=domain, seed=seed, budget=budget,candidate=cid,
                                        coefficient_max_abs_error=error, full_test_predictions_equal=True))
                print(f"REPLAY {path.name}",flush=True)
    result = dict(candidate_heads_replayed=checked, maximum_direct_solve_coefficient_error=largest,
                  independent_numpy_patch_extraction=True, validation_selection_verified=True,
                  complete_test_predictions_equal=True,records=records)
    write_json("reports/selection_revision/checkpoint_replay.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))


if __name__ == "__main__":
    with threadpool_limits(2):
        verify()
