# Changelog

## [0.5.0](https://github.com/timdewet/MycoMorph/compare/mycomorph-v0.4.0...mycomorph-v0.5.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* `mycomorph-cli label`, `review` and `train` moved to ModelPrep (`modelprep-cli label` / `review` / `train`, or the `modelprep` GUI), which reads the Features stage's all_crops.h5 instead of TIFF or labeled_data/ folders; the `napari` extra is removed.

### Features

* add a --modelprep flag that opens ModelPrep on its own ([6a1da3c](https://github.com/timdewet/MycoMorph/commit/6a1da3ca8e7c81ba9e33a2fdc9202ad13ec36cba))
* add modelprep and modelprep-cli commands ([57beb83](https://github.com/timdewet/MycoMorph/commit/57beb83e8591632c920755f3d9afaee9f2e760c6))
* list ModelPrep's trained runs in the Classify model picker ([72da127](https://github.com/timdewet/MycoMorph/commit/72da127a218441a98a3e8a45e65dfe72e612d1fe))
* ModelPrep lists presets from mycomorph/core/models ([b087626](https://github.com/timdewet/MycoMorph/commit/b08762607c160661c144c984fd4374a5eadde1b4))
* ModelPrep reads the crop schema MycoMorph writes ([7052b1a](https://github.com/timdewet/MycoMorph/commit/7052b1a97a312d747128b278ed50a6af3a896744))
* move label, review and train from mycomorph-cli to ModelPrep ([8c6fa40](https://github.com/timdewet/MycoMorph/commit/8c6fa40be763b37efa8bd852812c17a31ee6e923))
* open ModelPrep's window from MycoMorph's Labeller / trainer ([364ef50](https://github.com/timdewet/MycoMorph/commit/364ef503f81f3af765ea70861c8790d255cf0a48))
* point mycomorph-cli label, review and train to ModelPrep ([12f2761](https://github.com/timdewet/MycoMorph/commit/12f2761b04c45faa3e0237c7c86b4a3e58dfd1d3))
* version ModelPrep with MycoMorph ([246e7cf](https://github.com/timdewet/MycoMorph/commit/246e7cfd2cb1248dcc62390ccfee3570c70fcac9))


### Bug Fixes

* find the labelling rubric under docs/modelprep ([0509681](https://github.com/timdewet/MycoMorph/commit/0509681e79e293c0915cddc4954d79cb998924e9))
* say that segment --sample output doesn't feed ModelPrep ([74c9dc9](https://github.com/timdewet/MycoMorph/commit/74c9dc96177e699e1d4380ec218faec7df0dc29f))


### Performance Improvements

* read each cell's bounding box instead of scanning the whole mask ([26e40c9](https://github.com/timdewet/MycoMorph/commit/26e40c92fe7173efa58e349c9b37cab0ee17c103))


### Documentation

* cover ModelPrep, the macOS app and the bundle licence in the README ([10fb899](https://github.com/timdewet/MycoMorph/commit/10fb8998b1f1602063adf3b7c3d4c45693923560))
* fix the macOS first-launch and ModelPrep launch instructions ([7cf26a9](https://github.com/timdewet/MycoMorph/commit/7cf26a9e18b68f4a7ee1d4733ccf00176898d7ff))
* index ModelPrep's docs and map its pre-move paths ([7b0507e](https://github.com/timdewet/MycoMorph/commit/7b0507e2687036d321fd21fdd5986104960e99a7))
* point GPU users to the from-source install ([b2b59fe](https://github.com/timdewet/MycoMorph/commit/b2b59feb2a2d53f2d328e27f499f0509933edfa8))
* rewrite README to cover the current pipeline ([9fc45b6](https://github.com/timdewet/MycoMorph/commit/9fc45b60aede58f88d6deeaae4c31578c31d46a5))

## [0.4.0](https://github.com/timdewet/MycoMorph/compare/mycomorph-v0.3.0...mycomorph-v0.4.0) (2026-10-15)


### Features

* opt-in use of ModelPrep's tuned decision threshold in Classify ([6353906](https://github.com/timdewet/MycoMorph/commit/6353906242ab9ad72cd728cffdda464cbc9abb62))
* record crop options and Classify pre-filtering in the crops HDF5 ([f5a3d58](https://github.com/timdewet/MycoMorph/commit/f5a3d583f2e676137144637821e2f4c72c50d52c))
* record software versions in run outputs ([a2dd6a5](https://github.com/timdewet/MycoMorph/commit/a2dd6a547706d5fcd6b9d0901316fb542688693e))
* record software versions in run outputs ([737fe04](https://github.com/timdewet/MycoMorph/commit/737fe041af6295cd429255e7d53f45e6d9688597))


### Documentation

* name MycoMorph in the shared model block header ([1a47334](https://github.com/timdewet/MycoMorph/commit/1a47334fae23dd538ec3f80fd7c7439a89463097))

## [0.3.0](https://github.com/timdewet/MycoMorph/compare/mycomorph-v0.2.0...mycomorph-v0.3.0) (2026-08-06)


### Features

* in-app update checker and release-please automation ([e51ef80](https://github.com/timdewet/MycoMorph/commit/e51ef8087edc0fbd8fcbc08f7ee65ea65230248e))
* in-app update checker and release-please automation ([e2d4d19](https://github.com/timdewet/MycoMorph/commit/e2d4d19b4040d8c6a27ae9143e9f31afdf56bbc8))

## [0.2.0](https://github.com/timdewet/MycoMorph/compare/mycomorph-v0.1.0...mycomorph-v0.2.0) (2026-05-06)


### Features

* in-app update checker and release-please automation ([e51ef80](https://github.com/timdewet/MycoMorph/commit/e51ef8087edc0fbd8fcbc08f7ee65ea65230248e))
* in-app update checker and release-please automation ([e2d4d19](https://github.com/timdewet/MycoMorph/commit/e2d4d19b4040d8c6a27ae9143e9f31afdf56bbc8))
