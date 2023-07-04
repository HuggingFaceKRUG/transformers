<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer API를 사용한 하이퍼파라미터 검색 [[hyperparameter-search-using-trainer-api]]

🤗 Transformers는 훈련을 최적화한 [`Trainer`] 클래스를 제공하여, 직접 훈련 루프를 작성하지 않고도 🤗 Transformers 모델의 훈련을 쉽게 시작할 수 있습니다. [`Trainer`]는 하이퍼파라미터 검색을 위한 API도 제공합니다. 이 문서에서는 예제를 통해 그 사용 방법을 보여줍니다.

## 하이퍼파라미터 검색 백엔드 [[hyperparameter-search-backend]]

[`Trainer`]는 현재 네 가지 하이퍼파라미터 검색 백엔드를 지원합니다:
[optuna](https://optuna.org/), [sigopt](https://sigopt.com/), [raytune](https://docs.ray.io/en/latest/tune/index.html) 및 [wandb](https://wandb.ai/site/sweeps).

하이퍼파라미터 검색 백엔드를 사용하기 전에 해당 라이브러리를 설치해야 합니다.
```bash
pip install optuna/sigopt/wandb/ray[tune] 
```

## 예제에서 하이퍼파라미터 검색 활성화하는 방법 [[how-to-enable-hyperparameter-search-in-example]]

하이퍼파라미터 검색 공간을 정의하고, 백엔드에 따라 다른 형식이 필요합니다.

sigopt의 경우, sigopt [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter)를 참조하세요. 다음과 같은 형식입니다:
```py
>>> def sigopt_hp_space(trial):
...     return [
...         {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
...         {
...             "categorical_values": ["16", "32", "64", "128"],
...             "name": "per_device_train_batch_size",
...             "type": "categorical",
...         },
...     ]
```

optuna의 경우, optuna [object_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py)를 참조하세요. 다음과 같은 형식입니다:

```py
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

raytune의 경우, raytune [object_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html)를 참조하세요. 다음과 같은 형식입니다:

```py
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

wandb의 경우, wandb [object_parameter](https://docs.wandb.ai/guides/sweeps/configuration)를 참조하세요. 다음과 같은 형식입니다:

```py
>>> def wandb_hp_space(trial):
...     return {
...         "method": "random",
...         "metric": {"name": "objective", "goal": "minimize"},
...         "parameters": {
...             "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
...             "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
...         },
...     }
```

`model_init` 함수를 정의하고, 이를 [`Trainer`]에 전달합니다. 예시로서 다음과 같이 작성할 수 있습니다:
```py
>>> def model_init(trial):
...     return AutoModelForSequenceClassification.from_pretrained(
...         model_args.model_name_or_path,
...         from_tf=bool(".ckpt" in model_args.model_name_or_path),
...         config=config,
...         cache_dir=model_args.cache_dir,
...         revision=model_args.model_revision,
...         use_auth_token=True if model_args.use_auth_token else None,
...     )
```

`model_init` 함수, 훈련 인자, 훈련 및 테스트 데이터셋, 그리고 평가 함수를 사용하여 [`Trainer`]를 생성하세요:

```py
>>> trainer = Trainer(
...     model=None,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
...     tokenizer=tokenizer,
...     model_init=model_init,
...     data_collator=data_collator,
... )
```

하이퍼파라미터 검색을 호출하고, 최적의 시행(trial) 파라미터를 얻으세요. 백엔드는 `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"`일 수 있습니다. direction은 `"minimize"` 또는 `"maximize"`가 될 수 있으며, 목적함수를 최소화할지 최대화할지를 나타냅니다.

만약 직접 목적함수를 정의하고 싶다면, 정의하지 않았을 경우 기본 compute_objective 함수가 호출되며, F1과 같은 평가 지표의 합계가 목적값으로 반환됩니다.

```py
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## DDP finetune을 위한 하이퍼파라미터 검색 [[hyperparameter-search-for-ddp-finetune]]
현재로서, DDP(Distributed Data Parallel)에 대한 하이퍼파라미터 검색은 optuna와 sigopt에서 가능합니다. 검색 시행(trial)은 랭크-0 프로세스에서만 생성되고, 다른 랭크로 인수가 전달됩니다.