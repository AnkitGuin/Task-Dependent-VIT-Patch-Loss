# Third-party materials in the research release

Project software uses the repository's MIT license. This does not relicense third-party materials.

## DeiT-derived operator matrices

The official Facebook Research DeiT project distributes its code/pretrained-model repository under Apache License 2.0:

- Repository: https://github.com/facebookresearch/deit
- Retained license: `licenses/DeiT-Apache-2.0.txt`
- Original weight URL: https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
- Paper: Touvron et al., *Training data-efficient image transformers & distillation through attention*, ICML 2021.

**Modification notice:** evidence files named `operator_p16_deit_native_svd.npz` and `operator_p16_deit_rowspace.npz` contain modified derivatives of the original `patch_embed.proj.weight` matrix. We convert stored float32 values to float64, reshape the RGB patch map, perform an SVD, truncate the retained row dimension to the declared feature budget, and either retain its singular scaling or orthonormalize the retained rows. They are not full original checkpoints or complete DeiT encoders. Per-run metadata, source/configuration fingerprints and pretest records specify the transformation. The supplied upstream license applies to those derivatives; all existing upstream attribution/terms are retained here. The full downloaded weight file is excluded.

## Datasets and other assets

The historical supporting `vit_b16_patch_matrix.npz` contains a float64 reshape of the patch-projection layer from torchvision's `ViT_B_16_Weights.IMAGENET1K_V1`. It is a modified research extract, not a complete original checkpoint. The torchvision project's BSD 3-Clause license is retained in `licenses/Torchvision-BSD-3-Clause.txt`; source: https://github.com/pytorch/vision. The historical `deit_tiny_patch_matrix.npz` is likewise a reshaped research extract of the DeiT matrix and is covered by the DeiT attribution above. Full upstream weight URLs/checksums are in the asset manifest. Repository licenses do not grant rights to underlying training images or override upstream dataset terms.

CIFAR and SVHN images are not redistributed. Obtain them from the publishers using the checksum-verifying scripts and follow their terms. The SVHN publisher marks the downloads for non-commercial use. Official IDs, content hashes, aggregate observations, predictions and fitted research artifacts document the experiment; they do not grant a license to upstream datasets. No full ViT-B/16 pretrained model or dependency binaries are redistributed. Other scholarly references are citations, not claims of authorship of their methods.
