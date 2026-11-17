# TransUNet source attribution

`backend/transunet.py` is an inference adaptation of the official
[Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) implementation,
distributed under the Apache License 2.0. The complete upstream license is
included in `LICENSE` in this directory.

Upstream revision: `02ef0010b36eb8328b5e689eadaf613602edf9b8`.
Retrieved on 2026-11-13 from that exact GitHub revision. The `.upstream` files
are unmodified source snapshots kept for provenance; the app does not import
them or download source at runtime:

| Local snapshot | Upstream path | SHA-256 |
| --- | --- | --- |
| `vit_seg_modeling.py.upstream` | `networks/vit_seg_modeling.py` | `ab855c7dadb7af71a1ac09adba4d75c74a9867f89480c65eb9abb64461f3fa3f` |
| `vit_seg_modeling_resnet_skip.py.upstream` | `networks/vit_seg_modeling_resnet_skip.py` | `bb66abb598e38be27d738399cd2e99fe21844001045215f417fe606812df3ee5` |
| `vit_seg_configs.py.upstream` | `networks/vit_seg_configs.py` | `661639dec10545b2038d95a2414b76ea9552607613889cab2c25ef3ec1bc453c` |

Modifications in the adaptation: retained only the R50-ViT-B/16 forward
architecture, fixed the binary output head, removed training and NumPy
pretrained-weight conversion helpers, removed SciPy/ml_collections
dependencies, added input validation, inferred resolution from checkpoint
positions, and added strict loading with `base_model.` prefix handling and
meta-device construction. The original attention, weight standardization,
normalization, ResNet skip padding and decoder interpolation are preserved.

The supplied course checkpoint is separate from the upstream source license;
its authorship, training procedure, training normalization, and redistribution
terms were not supplied. This attribution does not claim ownership of it.
