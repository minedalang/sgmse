# Independent speaker probe (CAM++)

Used local WeSpeaker CAM++ (`avg_model.pt`) on 32 manifest utterances. `original_router` maps to the existing `rrsd_tcut_screen/router` directory. Mean enhanced-to-clean cosine: off 0.88595, original_router 0.88089, pair 0.88289, proto01 0.88286, proto01_cw1 0.88176.

Reader grouping yielded only 2 same-reader pairs and 494 different-reader pairs, so separation is diagnostic only. Same/different mean pair cosine (averaging utterance clean-cosines): off 0.85509/0.88607; original_router 0.85000/0.88102; pair 0.85232/0.88302; proto01 0.85184/0.88299; proto01_cw1 0.85035/0.88189. The sample has too few same-reader cross-utterance pairs to claim speaker verification improvement.
