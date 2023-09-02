# IncentiveAllocation
## Evaluator
### BinaryTreatmentEvaluator
#### BaseBinaryTreatmentEvaluator
- 二値の介入における評価値計算の抽象クラス
  - オブジェクトの初期化をする .__init__()メソッドを持つ
    - 入力：以下をインプットとしする
      - t_opt: 介入最適化結果 (np.ndarray, shape=(N,)),
      - t_test: 介入実績 (np.ndarray, shape=(N,)),
      - y_test: 目的変数実績 (np.ndarray, shape=(N,)),
      - c_test: コスト実績 (np.ndarray, shape=(N,)),
      - p_test: 介入確率実績 (np.ndarray, shape=(N,), optional),
    - 計算：
      - 上記のインプットを全て同じ変数名でメンバ変数として格納する
    - 出力：
      - 何も出力しない 
  - モデルの評価値を計算する .score() メソッドをもつ 
    - 入力：以下をインプットとする
      - budget: 予算 (int, optional)
    - 計算：
      - __init__() で定義した介入最適化結果・介入(+介入確率)・目的変数・コストの実績を照らし合わせて評価を行う（ただし、BaseBinaryTreatmentEvaluatorでは実装をパスしてエラーを返す）
    - 出力：
      - 処理2で計算したスコアを出力する(float)
  - 予算帯ごとの評価値を計算する .score_curve() メソッドを持つ
    - 入力：以下をインプットとする
      - budget_array: 予算配列 (np.ndarray, optional)
      - budget_rate_array: 予算比率配列 (np.ndarray, optional)
      - n_split: 分割数
    - 計算：
      - budget_arrayが入力されていれば、その配列の各要素を予算として設定した場合のの評価値を .score()メソッドを使って計算し、budget_arrayに対応したスコアの配列を得る
      - budget_rate_arrayが入力されていれば、ランダム配布時のコストにbudget_rate_arrayを掛け算してbudget_arrayを作成する（それ以降は同様）
      - n_split が入力されていればランダム配布時のコストをn_splitだけbinにとってbudget_arrayを作成する（それ以降は同様）
      - ただし、budget_array, budget_rate_array, n_split の複数が入力されている場合、エラーを吐いて終了する
    - 出力：
      -  budget_rate_arrayおよびbudget_arrayに対応した配列を返却する



#### AddonPerCost
- BaseBinaryTreatmentEvaluatorを継承して作成する、モデルのコストパフォーマンスを評価する評価値計算象クラス
  - .score() メソッドをオーバーライドし、処理2を以下の通りに変更する
    - 処理2：
      - 処理2-1: self.t_opt が 1となるユーザーのみについて、目的変数実績からアドオンを算出する
      - 処理2-2: self.t_opt が 1となるユーザーのみについて、実績のコストの合計を算出する
      - 処理2-3: 処理2-1の結果を処理2-2の結果で割った値を評価値とみなす

#### Addon
- BaseBinaryTreatmentEvaluatorを継承する、モデルのアドオン総量を評価する評価値計算象クラス
  - .score() メソッドをオーバーライドし、処理2を以下の通りに変更する
    - 処理2: self.action が 1となるユーザーのみについて、目的変数実績からアドオンを算出し、その値を評価値とみなす



####　score_curve_AUC
- 複数のモデルについてBaseBinaryTreatmentEvaluatorクラスの.score_curve()関数が算出する結果を可視化する関数
  - 入力：以下をインプットとする
    - file_name: グラフを保存するファイル名 (Path or string)
    - show: グラフを表示するか否か (binary)
    - result_array: 以下のような辞書 (dict)
      {
        "model1": {"budget_rate_array": np.ndarray, "score_array": np.ndarray},
        "model2": {"budget_rate_array": np.ndarray, "score_array": np.ndarray}
        ...
      }
  - 計算:
      - 処理1: 以下のパターンで実行する
        - file_name is not null & show=True: 各モデルのscore_curveを描画して、画像をfileに保存して終了
        - file_name is not null & show=False: 各モデルのscore_curveを描画せず、画像をfileに保存して終了
        - file_name is null & show=True: 各モデルのscore_curveを描画して、fileに保存せず終了
        - file_name is null & show=False:何もしないで終了
      - 処理2: 各モデルのscore_curveに対するAUCを計算する
  - 出力：
      - 各モデルのscore_curveに対するAUCを以下のような辞書形式で返却
      {"model1": float, model2": float, ...}

### MultipleTreatmentEvaluator
- TBD

## Policy
### BinaryTreatmentPolicy
#### BaseBinaryTreatmentPolicy
- 二値の介入における意思決定モデル
  - オブジェクトの初期化をする .__init__()メソッドを持つ
    - 具体的なLearner および Optimizer をそれぞれself.learner, self.optimizerに設定する
  - 機械学習モデルを学習させる .fit() メソッドをもつ 
    - 入力：以下をインプットとする
      - X_train: 過去の実績結果における各ユーザーの特徴量 (np.ndarray, shape=(N, M)),
      - y_train: 過去の実績結果における各ユーザーの目的変数 (np.ndarray, shape=(N,)),
      - c_train: 過去の実績結果における各ユーザーの消費コスト (np.ndarray, shape=(N,)),
      - t_train: 過去の実績結果における各ユーザーの介入状況  (np.ndarray, shape=(N,)),
      - p_train: 過去の実績結果における各ユーザーの介入確率 (np.ndarray, shape=(N,), optional),
    - 計算：
      - __init__() 関数で設定したLearnerにインプットを入力して学習を行う
    - 出力：
      - 何も出力しない
  - 学習した機械学習モデルの推論結果を算出する .predict() メソッドを持つ
    - 入力：以下をインプットとする
      - X_test: 最適化対象の各ユーザーの特徴量 (np.ndarray, shape=(N, M)),
    - 計算：
      - fit関数で学習したモデルにX_testを入力し、各機械学習モデルの推論結果を計算する
    - 出力：
      - 各機械学習モデルにおいて、モデルの役割をkeyにその推論結果の配列をvalueに持つ辞書を返却する({"string": np.ndarray})
  - 学習した機械学習モデルの推論結果を用いて割り当てを計算する .optimize() メソッドを持つ
    - 入力：以下をインプットとする
      - X_test: 最適化対象の各ユーザーの特徴量 (np.ndarray, shape=(N, M)),
    - 計算：
      - X_testを入力に .predict() 関数を実行し、推論結果を得る
      - 推論結果を .__init__()関数で設定した self.optimizerのrun()メソッドに入力して最適割り当てを得る
    - 出力：
      - 最適な介入結果を配列として返す (np.ndarray, shape=(N,)),

#### OneStagePolicy
- BaseBinaryTreatmentPolicyを継承して作成する、1段階で直接付与優先度を決定し、介入を最適化をするモデル
  - 

#### TwoStagePolicy
- BaseBinaryTreatmentPolicyを継承して作成する、1段階目で目的変数およびコストに対する予測・2段階目で介入の最適化をするモデル
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 入力：以下をインプットとする
      - addon_learner: 　(Learner.BinaryTreatmentLearner)
      - cost_leaner: (sklearn APIの機械学習モデル)
      - optimizer: (Optimizer.BinaryTreatmentOptimizer)
    - 計算: 
      - 処理1: 入力値を同じ変数名でメンバ変数として格納する
      - 処理2: addon_learnerに設定したLearnerのメンバ変数 .valid_policy に"addon"が含まれていない場合エラーで落とす
  - 機械学習モデルを学習させる .fit() メソッドをもつ 
    - 計算：
      - 処理1: X_train, y_train, t_train, p_train をself.addon_learnerのfitメソッドに入力
      - 処理2: X_train, c_train, t_train, p_train をself.cost_learnerのfitメソッドに入力
    - 出力：
      - 何も出力しない
    - .predict()関数をオーバーライドして以下の通りに変更する
      - 入力：以下をインプットとする
        - X_test: 最適化対象の各ユーザーの特徴量 (np.ndarray, shape=(N, M)),
      - 計算：
        - self.fit()で学習したaddon_leanerモデルおよびcost_lernerにX_testを入力し、推論結果を得る
      - 出力：
        - addon_leaner、cost_lernerの結果をまとめ、以下のフォーマットの辞書を返却する
        ({"addon": np.ndarray, "cost": np.ndarray})



## Learner
### BinaryTreatmentLearner
#### BaseBinaryTreatmentLearner
- オブジェクトの初期化をする .__init__()メソッドを持つ
    - 入力：以下をインプットとする
      - base_learner: sklearn APIの機械学習モデルのインスタンス
    - 計算：
      - 入力値を同じ変数名でメンバ変数として格納する
    - 出力：
      - 何も出力しない 
  - 機械学習モデルを学習させる .fit() メソッドをもつ 
    - 入力：以下をインプットとする
      - X_train: 過去の実績結果における各ユーザーの特徴量 (np.ndarray, shape=(N, M)),
      - y_train: 過去の実績結果における各ユーザーの目的変数 (np.ndarray, shape=(N,)),
      - t_train: 過去の実績結果における各ユーザーの介入状況  (np.ndarray, shape=(N,)),
      - p_train: 過去の実績結果における各ユーザーの介入確率 (np.ndarray, shape=(N,), optional),
    - 計算：
      - 機会学習モデルの学習を行う。ただし、BaseBinaryTreatmentLearnerでは実装をパスしてエラーを返す
    - 出力：何も出力しない 
  - 学習した機械学習モデルの推論結果を算出する .predict() メソッドを持つ
    - 入力：以下をインプットとする
      - X_test: 最適化対象の各ユーザーの特徴量 (np.ndarray, shape=(N, M)),
    - 計算：
      - 機会学習モデルの推論を行う。ただし、BaseBinaryTreatmentLearnerでは実装をパスしてエラーを返す
    - 出力：
      - 各モデルが推論した結果を辞書形式で返却する。ただし、BaseBinaryTreatmentLearnerでは実装をパスしてエラーを返す。

#### TLearner
- BaseBinaryTreatmentLearnerを継承して作成する、T-LearnerによるUplifit Modelingを実装するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["addon"] として設定する

  - .fit()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: p_trainが None の場合、p_trainを全ての要素が１で長さがy_trainと同じ配列として代入する 
      - 処理2: self.base_learner を二つのインスタンス self.tg_learner, self.cg_leanerとしてコピーする
      - 処理3: X_trainを特徴量、y_trainを目的変数、t_trainを介入実績, p_trainを傾向スコアとしてself.tg_learner, self.cg_leanerを用いてT-Learnerを作成する

  - .predict()関数をオーバーライドして以下の通りに変更する
    - 計算：
      - 処理1: self.tg_modelで予測した結果をy_tg_predとする
      - 処理2: self.cg_modelで予測した結果をy_cg_predとする
      - 処理3: y_tg_pred - y_cg_predの結果を y_addon_predとする
    - 出力:
      - {"TG": y_tg_pred, "CG": y_cg_pred, "addon":  y_addon_pred}として辞書を出力する
    
  

#### SLearner
- BaseBinaryTreatmentLearnerを継承して作成する、S-LearnerによるUplifit Modelingを実装するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["addon"] として設定する

  - .fit()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: p_trainが None の場合、p_trainを全ての要素が１で長さがy_trainと同じ配列として代入する 
      - 処理2: X_trainを特徴量、y_trainを目的変数、t_trainを介入実績, p_trainを傾向スコアとしてself.base_learnerを用いてS-Learnerを作成する

  - .predict()関数をオーバーライドして以下の通りに変更する
    - 計算：
      - 処理1: self.base_learnerを用いてtgの予測した結果をy_tg_predとする
      - 処理2: self.base_learnerを用いてcgの予測した結果をy_tg_predとする
      - 処理3: y_tg_pred - y_cg_predの結果を y_addon_predとする
    - 出力:
      - {"TG": y_tg_pred, "CG": y_cg_pred, "addon":  y_addon_pred}として辞書を出力する

#### TOTLeaner
- BaseBinaryTreatmentLearnerを継承して作成する、Transformed OutcomeによるUplifit Modelingを実装するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["addon"] として設定する
  - .fit()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: p_trainが None の場合、p_trainを全ての要素が１で長さがy_trainと同じ配列として代入する 
      - 処理2: X_trainを特徴量、y_trainを目的変数、t_trainを介入実績, p_trainを傾向スコアとしてself.base_learnerを用いてTransformed Outcomeの理論をもとにしたアップリフトモデリングを学習する

  - .predict()関数をオーバーライドして以下の通りに変更する
    - 計算：
      - 処理1: self.base_learnerを用いて予測したアップリフトの結果をy_addon_predとする
    - 出力:
      - {"addon":  y_addon_pred}として辞書を出力する

#### ROILeaner
- BaseBinaryTreatmentLearnerを継承して作成する、ROIを直接予測するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["score"] として設定する

#### FunnelROILeaner
- BaseBinaryTreatmentLearnerを継承して作成する、ファネルを仮定した状況でのROIを直接予測するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["score"] として設定する

#### FunnelReductionROILeaner
- BaseBinaryTreatmentLearnerを継承して作成する、ファネルを仮定し、かつファネルの１段階目がコスト消費を意味する状況でのROIを直接予測するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["score"] として設定する


#### CostUnawarePolicyGradientLeaner
- BaseBinaryTreatmentLearnerを継承して作成する、コストを考慮しない方策勾配法を実装するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["score"] として設定する

#### CostAwarePolicyGradientLeaner
- BaseBinaryTreatmentLearnerを継承して作成する、コストを考慮する方策勾配法を実装するクラス
  - .__init__()関数をオーバーライドして以下の通りに変更する
    - 計算:
      - 処理1: 親クラスの.__init__()関数を用いて、入力値を同じ変数名でメンバ変数として格納する
      - 処理2: self.valid_policy = ["score"] として設定する


## Optimizer
### BinaryTreatmentOptimizer
#### BaseBinaryTreatmentOptimizer

- オブジェクトの初期化をする .__init__()メソッドを持つ
    - 入力：以下をインプットとしする
    - 計算：
    - 出力：
      - 何も出力しない 
  - 割り当ての最適化を実行する .run() メソッドをもつ 
    - 入力：以下をインプットとする
    - 計算：
    - 出力：

#### ScoreGreedyOptimizer
- BaseBinaryTreatmentOptimizerを継承して作成する、予測付与優先度と予測コストをもとに割り当てを最適化するクラス
- stochastic, determinisiticに対応

#### AddonGreedyOptimizer
- BaseBinaryTreatmentOptimizerを継承して作成する、予測アドオンと予測コストをもとに割り当てを最適化するクラス
- stochastic, determinisiticに対応


# 想定パターン

- ルールベース
  - Policy: OneStagePolicy, TwoStagePolicyのどちらでも可
  - Leaner: BinaryTreatmentLearnerを継承して、.fitは何もせず、.predict()を自作のルールとして書く 
  - Optimizer: ScoreGreedyOptimizer, AddonGreedyOptimizerのどちらでも可

- コスト無視S-Leaner
  - Policy: TwoStagePolicy
  - Leaner: (addon_leaner) SLearner
  - Optimizer: AddonGreedyOptimizer

- コスト考慮S-Leaner
  - Policy: TwoStagePolicy
  - Leaner: (addon_leaner) SLearner
  - Optimizer: ScoreGreedyOptimizer

- コスト無視T-Leaner
  - Policy: TwoStagePolicy
  - Leaner: (addon_leaner) TLearner
  - Optimizer: AddonGreedyOptimizer

- コスト考慮T-Leaner
  - Policy: TwoStagePolicy
  - Leaner: (addon_leaner) TLearner
  - Optimizer: ScoreGreedyOptimizer

- コスト無視TOT
  - Policy: TwoStagePolicy
  - Leaner: (addon_leaner) TOTLearner
  - Optimizer: AddonGreedyOptimizer

- コスト考慮TOT
  - Policy: TwoStagePolicy
  - Leaner: (addon_leaner) TOTLearner
  - Optimizer: ScoreGreedyOptimizer

- ROIモデル
  - Policy: OneStagePolicy
  - Leaner: (score_leaner) ROILearner
  - Optimizer: ScoreGreedyOptimizer

- ファネルROIモデル
  - Policy: OneStagePolicy
  - Leaner: (score_leaner) FunnelROILeaner
  - Optimizer: ScoreGreedyOptimizer

- ファネル約分ROIモデル
  - Policy: 
  - Leaner: (score_leaner) FunnelReductionROILeaner
  - Optimizer: ScoreGreedyOptimizer

- コスト無視方策勾配
  - Policy: OneStagePolicy
  - Leaner: (score_leaner) CostUnawarePolicyGradientLeaner
  - Optimizer: AddonGreedyOptimizer

- コスト考慮方策勾配
  - Policy: OneStagePolicy
  - Leaner: (score_leaner) CostAwarePolicyGradientLeaner
  - Optimizer: ScoreGreedyOptimizer



# 利用例(S-Learnerで２段階ポリシーを採用し、個人のAddonPerCost順に)

```
# 各種機械学習モデルを定義
revenue_tg_model = LightGBMRegressor()
revenue_cg_model = LightGBMRegressor()
cost_tg_model = LightGBMClassifier()
t_leaner = TLearner(revenue_tg_model, revenue_cg_model, cost_tg_model)

# 最適化モデルを定義
greedy_optimizer = ScoreGreedyOptimizer()

# 付与ポリシーを定義
policy = BinaryTreatmentPolicy(leaner = t_leaner, optimizer = greedy_optimizer)

# 評価方法を定義
evaluator = AddonPerCost()

# 過去の実績データを用いて学習
policy.fit(X_train, y_train, c_train, t_train, p_train)

# 現在のデータを用いて割り当ての最適化を実行
t_opt, p_opt = policy.optimize(X_test)

# 評価を得る
score_curve = evaluator.score_curve(t_opt, t_test, y_test, c_test, p_test)
```