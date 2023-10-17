# ft-pipeline
My Fine-Tuning Pipeline in preperation for [this Kaggle](https://www.kaggle.com/competitions/contradictory-my-dear-watson)


Two notebooks that would be helpful for now:

[Finetune notebook](https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-train)
> You can extract this to be much more concise; you should understand what's going on in  prepare_inputs, collate, train_fn (ignore FGM and AWP for now), MeanPooling, FB3TrainDataset, FB3Model (ignore the __init__ ), and get_optimizer_params within train_loop. This is all the lego pieces you need for the pipeline.

[Utilizing Transformer Representations Efficiently](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently)
Encyclopedia of pooling; to be honest, I've never found success beyond CLS pooling and mean pooling, but this notebook is so well-made that it'd be a crime not share it. Lots of pictures to illustrate what's going on.

The dataset is multilingual, so a multilingual model is needed: I'd go with `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large` for now. Set `max_length = 512`.

Try to set up your fine-tuning pipeline and a 4-fold cross-validation (just random is fine now).
- Our dataset overlaps xnli and mnli, that's how perfect accuracy is achieved on the leaderboard.
- Feed `f"{premise} [SEP] {hypothesis}"` to the tokenizer.
- Use `P100` for GPU; if you want to explore distributed training with the two `T4`, look into [Accelerate](https://huggingface.co/docs/accelerate/index).
- I actually think CLS pooling will work better here, but it's good to learn what mean pooling is anyways.