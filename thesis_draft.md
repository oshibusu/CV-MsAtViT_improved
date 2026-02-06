```latex
\section{アーキテクチャの改良}
本研究では、先行研究で用いられていたSplit-Complexアプローチの限界に対処するため、完全な複素数対応アーキテクチャ（Fully Complex-valued Architecture）を提案する。我々は、ベースラインモデル（CV-MsAtViT Default）において位相情報が破棄または歪められていた複数のボトルネックを特定し、PolSARデータの代数的構造を厳密に保存するための数学的改良を導入した。主な変更点を表\ref{tab:arch_comparison}に要約する。

\begin{table}[htbp]
    \centering
    \caption{ベースラインと提案手法のアーキテクチャ比較}
    \begin{tabular}{l|p{6cm}|p{6cm}}
        \toprule
        \textbf{コンポーネント} & \textbf{ベースライン (Split-Complex)} & \textbf{提案手法 (Fully-Complex)} \\
        \midrule
        \textbf{Attention機構} & Split Attention (実部/虚部独立) & Complex Multi-Head Attention ($Q \cdot K^H$) \\
        \textbf{QKVバイアス} & 有効 (Keras標準) & \textbf{無効} (回転・スケーリング等価性の保存) \\
        \textbf{正規化} & Split Layer Normalization & Complex Layer Normalization (共分散ベース) \\
        \textbf{活性化関数 (ViT)} & Cartesian GELU & Modified ReLU (振幅ゲーティング) \\
        \textbf{CoordAttゲーティング} & Cartesian Sigmoid & Modified Tanh (位相保存型) \\
        \textbf{出力層} & 絶対値に対するSoftmax & 実部に対するSoftmax \\
        \bottomrule
    \end{tabular}
    \label{tab:arch_comparison}
\end{table}

\subsection{Complex Multi-Head Attention}
ベースラインモデルは、実部と虚部を独立した実数値Attentionヘッドで処理する単純なSplit-Complex Attention機構を採用していた。
\begin{equation}
    \text{Attention}(Q, K, V) \approx \text{MHA}(\Re(Q), \Re(K), \Re(V)) + j \cdot \text{MHA}(\Im(Q), \Im(K), \Im(V))
\end{equation}
このアプローチは、位相差のモデリングに不可欠なクロス項（例：$\Re(Q)\cdot\Im(K)$）を無視している。
対照的に、提案手法では厳密な複素数値Attention機構を実装した。
\begin{equation}
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\Re(Q K^H)}{\sqrt{d_k}}\right) V
\end{equation}
ここで、$K^H$は共役転置を表す。さらに、$Q, K, V$の線形射影においてバイアス項を明示的に無効化することで、Attention操作が複素領域における純粋な線形変換であることを保証し、学習されたバイアスベクトルによって引き起こされる不当な位相シフトを回避した。

\subsection{位相保存型 Coordinate Attention}
Coordinate Attention (CA) ブロックにおいて、ベースラインはAttentionマップに対して以下のCartesian Sigmoid活性化を使用していた。
\begin{equation}
    \sigma_{cart}(z) = \text{sigmoid}(\Re(z)) + j \cdot \text{sigmoid}(\Im(z))
\end{equation}
これは実部と虚部に独立して実数値シグモイド関数を適用するものであり、Attention重みの位相情報を歪めてしまう。
我々はこれを、以下のように定義されるModified Tanh (ModTanh) または Modified Sigmoid活性化に置き換えた。
\begin{equation}
    \sigma_{mod}(z) = \frac{1.0 + \text{tanh}(|z| + b)}{2} \cdot \frac{z}{|z| + \epsilon}
\end{equation}
これにより、ゲーティング機構が特徴量の振幅のみに作用し、元の位相情報を保存することを保証した。

\subsection{複素正規化と活性化関数}
ベースラインモデルは、複素信号を2つの独立した実数チャネルとして扱うSplit Layer NormalizationとCartesian活性化（CartReLU/CartGELU）に依存していた。
我々は、実部と虚部の分散の和である「合計分散（Total Variance）」を用いて正規化を行うComplex Layer Normalizationを採用した。また、活性化関数には、振幅に対して閾値処理を行うModified ReLU (ModReLU) を用いた。
\begin{equation}
    \sigma^2 = \text{Var}(\Re(z)) + \text{Var}(\Im(z))
\end{equation}
\begin{equation}
    \hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}
\end{equation}
\begin{equation}
    \text{ModReLU}(z) = \text{ReLU}(|z| + b) \cdot \frac{z}{|z| + \epsilon}
\end{equation}
これらの変更により、ネットワークの深層に至るまでPolSARコヒーレンシ行列の全体的な性質が維持される。
最後に、分類ヘッドにおいて、ベースラインは複素ロジットの絶対値に基づいて確率を計算する \texttt{softmax\_real\_with\_abs} を使用しており、干渉解析に不可欠な位相情報を破棄していた。提案手法では、複素ロジットの実部を利用する \texttt{softmax\_real\_with\_real} を提案した。これは、コヒーレンシ行列の対角要素の物理的解釈と整合するように、実部を信頼度スコアとして解釈するものである。

\subsection{実装の安定性}
また、ベースライン実装における数値的不安定性にも対処した。具体的には、元のライブラリにおけるComplex Batch Normalization層の初期化がモデルのシリアライズ時に型エラーを引き起こしていた。我々は、厳密な型安全性を確保し、安定したモデルのチェックポイント保存とデプロイを可能にするため、\texttt{ForceFloatInitializer} を備えたラッパークラス \texttt{FixedComplexBatchNormalization} を導入した。


\section{Experimental Results}
In this section, we present the experimental results on three PolSAR datasets: Flevoland, San Francisco, and Baltrum. We compare the proposed method with the baseline (CV-MsAtViT Default) both qualitatively and quantitatively.

\subsection{Visual Comparison}

\subsubsection{Flevoland Dataset}
Figure \ref{fig:fl_t_maps} shows the classification results for the Flevoland dataset.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figs/FL_T_comparison_grid.png}
  \caption{Visual comparison on Flevoland dataset. Top-Left: Pauli RGB, Top-Right: Ground Truth, Bottom-Left: Baseline (Default), Bottom-Right: Proposed Method.}
  \label{fig:fl_t_maps}
\end{figure}

\subsubsection{San Francisco Dataset}
Figure \ref{fig:sf_maps} presents the results for the San Francisco dataset.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figs/SF_comparison_grid.png}
  \caption{Visual comparison on San Francisco dataset. Top-Left: Pauli RGB, Top-Right: Ground Truth, Bottom-Left: Baseline (Default), Bottom-Right: Proposed Method.}
  \label{fig:sf_maps}
\end{figure}

\subsubsection{Baltrum Dataset (FP1 S-band)}
Figure \ref{fig:baltrum_maps} illustrates the results for the Baltrum Island dataset. The input PolSAR data (Pauli RGB) is shown in (a), followed by the ground truth and prediction maps.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figs/Baltrum_comparison_grid.png}
  \caption{Visual comparison on Baltrum dataset (FP1, S-band). Top-Left: Pauli RGB, Top-Right: Ground Truth, Bottom-Left: Baseline (Default), Bottom-Right: Proposed Method.}
  \label{fig:baltrum_maps}
\end{figure}

\subsection{Learning Curves}
Figure \ref{fig:learning_curves} displays the training accuracy and loss curves for each dataset.

\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.32\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/default_FL_T_learning_curve.png}
    \caption{Flevoland (Default)}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.32\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/default_sf_learning_curve.png}
    \caption{San Francisco (Default)}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.32\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/default_baltrum_learning_curve.png}
    \caption{Baltrum (Default)}
  \end{subfigure}
  
  \vspace{1em}
  
  \begin{subfigure}[b]{0.32\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/propose_FL_T_learning_curve.png}
    \caption{Flevoland (Proposed)}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.32\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/propose_SF_learning_curve.png}
    \caption{San Francisco (Proposed)}
  \end{subfigure}
  % Baltrum Proposed curve if available similarly
  \caption{Comparison of learning curves between Baseline and Proposed method.}
  \label{fig:learning_curves}
\end{figure}

\subsection{Quantitative Analysis}
To further analyze the classification performance, we present the confusion matrices in Figure \ref{fig:confusion_matrices}.

\begin{figure}[htbp]
  \centering
  % Flevoland
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/default_CV_MsAtViT_FL_T_cm.png}
    \caption{Flevoland (Baseline)}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/propose_CV_MsAtViT_FL_T_cm.png}
    \caption{Flevoland (Proposed)}
  \end{subfigure}
  
  \vspace{1em}

  % San Francisco
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/default_CV_MsAtViT_SF_cm.png}
    \caption{San Francisco (Baseline)}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/propose_CV_MsAtViT_SF_cm.png}
    \caption{San Francisco (Proposed)}
  \end{subfigure}

  \vspace{1em}

  % Baltrum
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/CV_MsAtViT_default_Baltrum_cm.png}
    \caption{Baltrum (Baseline)}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figs/CV_MsAtViT_propose_Baltrum_S_FP1_cm.png}
    \caption{Baltrum (Proposed)}
  \end{subfigure}

  \caption{Confusion matrices for the three datasets. The rows represent true labels and columns represent predicted labels.}
  \label{fig:confusion_matrices}
\end{figure}


\subsection{結果の詳細分析}

\subsubsection{学習曲線 (Learning Curves)}
各データセットにおける学習の収束挙動を確認した結果、以下の傾向が観察された。

\begin{itemize}
    \item \textbf{Flevoland (FL)}:
    \begin{itemize}
        \item \textbf{Baseline}: 学習初期から損失(Loss)が滑らかに減少し、精度(Accuracy)も単調に向上する安定した収束挙動を示した。最終的なTraining Accuracyは98\%以上に達し、非常に安定している。
        \item \textbf{Proposed}: 学習初期（最初の5エポック程度）において損失値の変動（振動）が見られたが、その後急速に収束し、最終的にはBaselineと同等の98\%付近の高い学習精度を達成した。初期の不安定さはあるものの、最終的な到達点は同等である。
    \end{itemize}
    \item \textbf{San Francisco (SF) / Baltrum}:
    \begin{itemize}
        \item 同様に、Baselineは滑らかな収束を示す一方、Proposed手法も十分な収束性を示しているが、データセットの複雑さに応じて収束速度や最終到達精度に若干の差異が見られた。
    \end{itemize}
\end{itemize}

\subsubsection{各クラスの学習結果 (Classification Report)}
各クラスごとの適合率(Precision)、再現率(Recall)を比較分析した結果、提案手法とベースラインで明確な特性の差が確認された。

\begin{itemize}
    \item \textbf{Flevoland Dataset}:
    \begin{itemize}
        \item \textbf{全体傾向}: 多くのクラス（Water, Forest, Wheat等）において、両手法ともに95\%以上の高いF1スコアを達成した。
        \item \textbf{提案手法の課題}: 「Bare Soil (裸地)」クラスにおいて、BaselineがRecall 99.56\%を達成しているのに対し、提案手法はRecall 27.02\%と著しく低い値となった。これは提案手法がBare Soilを他の植生クラス（特にBarley）と誤認していることに起因する。
        \item \textbf{提案手法の利点}: 一方で、「Lucerne」などの一部の植生クラスにおいては、Baselineよりも高いRecallないしPrecisionを示すケースも見られた。
    \end{itemize}

    \item \textbf{San Francisco Dataset}:
    \begin{itemize}
        \item \textbf{全体傾向}: 両手法ともにOverall Accuracy 97\%越えの高い性能を示した。
        \item \textbf{クラス別}: 「Water」や「Urban」といった主要クラスでは、両手法ともにF1スコア98\%以上と極めて高精度である。提案手法は「Vegetation」クラスの識別において、Baseline（F1 0.90）に比べ若干劣る（F1 0.88）結果となったが、全体的な性能差は僅差である。
    \end{itemize}

    \item \textbf{Baltrum Dataset}:
    \begin{itemize}
        \item \textbf{全体傾向}: 非常に複雑な自然環境を含む本データセットでは、Baseline (OA 89.42\%) が提案手法 (OA 87.24\%) をわずかに上回った。
        \item \textbf{提案手法の課題}: 特に「Coastal shrub (低木)」と「White dune (白砂丘)」の識別において苦戦が見られた。White duneのRecallはBaselineが82.11\%であるのに対し、提案手法は52.71\%に留まった。
        \item \textbf{提案手法の利点}: 「Water (水域)」クラスに関しては、提案手法（Recall 98.14\%）がBaseline（93.54\%）を上回る識別能力を示しており、水域と陸域の境界判定において優位性が見られた。
    \end{itemize}
\end{itemize}

\subsubsection{混同行列 (Confusion Matrix)}
誤分類の具体的な内訳（クラス間の混同）を分析した。

\begin{itemize}
    \item \textbf{Flevoland}:
    \begin{itemize}
        \item \textbf{Proposed}: 最大の誤分類は「Bare Soil」$\to$「Barley (大麦)」への誤認であり、Bare Soilサンプルの71\%がBarleyとして予測されていた。これは実数値のみに基づくBaselineでは発生していない特異な混同パターンである。
    \end{itemize}
    \item \textbf{Baltrum}:
    \begin{itemize}
        \item \textbf{Proposed}: 「White dune」のサンプルの約43\%が「Grey dunes (灰色砂丘)」として誤分類されており、これが精度の低下の主要因となっている。また、「Coastal shrub」も「Dense, high vegetation」や「Grey dunes」と混同される傾向が強く、植生密度や地表面の材質が類似するクラス間での分離に課題を残した。
        \item \textbf{Baseline}: Baselineも同様の傾向を示すが、Proposedに比べて混同の割合は低く抑えられており、類似クラス間の境界をよりロバストに捉えていることが示唆される。
    \end{itemize}
\end{itemize}


\section{考察}

本研究では、PolSARデータの物理的性質（位相情報）を保存するために完全な複素数アーキテクチャを導入した。実験結果に見られるBaselineとの差異について、モデルの構造的観点から以下の考察を行う。

\subsection{モデルの表現能力とパラメータ数のトレードオフ}
本提案手法では、Attention機構や活性化関数において、従来の実部・虚部独立（Split-Complex）な計算を、数学的に厳密な複素演算（Complex-Valued）に置き換えた。
これにより、モデルは信号の位相を正しく保持することが可能となった一方で、学習可能なパラメータの自由度（Effective Degrees of Freedom）は減少している。
先行研究のSplit-Complexアプローチでは、実部と虚部がそれぞれ独立した重みパラメータによって最適化されるため、物理的な位相整合性を無視した柔軟な（ある種、過剰な）フィッティングが可能であった。対して、提案手法における複素行列積は、実部と虚部の重み間に「回転とスケーリング」という厳格な代数的制約を課すことと同義であり、実質的なパラメータ数が減少したと解釈できる。
実験結果において、提案手法がBaselineに比べて収束に時間を要したり、一部の複雑なクラス（Baltrumデータセット等）で精度が低下した要因として、このパラメータ数の減少による学習容量（Model Capacity）の低下が、位相保存による情報量のメリットを上回ってしまった可能性が考えられる。

\subsection{ModReLUのスパース性と残差学習への影響}
また、活性化関数としてのModReLUと、バックボーンであるResNet（残差結合）構造との相性についても考察が必要である。
本研究で採用したModReLUは、複素数の振幅に対してReLUを適用し、位相を保存する関数である。
\begin{equation}
 \text{ModReLU}(z) = \text{ReLU}(|z| + b) \cdot \frac{z}{|z| + \epsilon}
\end{equation}
比較対象であるCartesian ReLUは、入力の実部・虚部それぞれの正負に応じて独立に値を0にするため、出力信号は高いスパース性（疎性）を持つ。スパースな活性化は、Deep Learning、特にResNetにおいて「必要な特徴のみを残差として加算し、不要な情報は0にする」メカニズムを機能させる上で重要である。
しかし、ModReLUは振幅が一様なしきい値を超えている限り、位相の全方向に対して値を出力するため、Cartesian ReLUに比べて出力のスパース性が著しく低下する傾向がある。
この「スパース性の欠如」は、残差ブロックが恒等写像（Identity Mapping）を効率的に学習することを阻害し、結果として深層特徴の表現力を制限してしまった可能性がある。
したがって、位相を保存しつつ適切なスパース性を確保できるようなバイアスの初期化戦略や、新たな複素活性化関数の検討が今後の課題である。

\subsection{特徴抽出部の簡素化と将来の展望}
さらに、本提案手法のようにネットワーク全体で位相情報を厳密に保存・伝播させる場合、初段の特徴抽出器（Multi-Scale Feature Extractor）として計算コストの高い複素CNN（CV-CNN）を用いる必然性が薄れる可能性がある。
従来のSplit-Complexモデルでは、初期段階で位相情報を可能な限り「振幅のパターン」として空間特徴量に変換する必要があったが、Fully Complexモデルでは生の位相情報をそのまま後段のTransformer層まで損失なく伝達できるためである。
したがって、初段の3次元CNNをより簡易的な構造、例えば複素全結合層（Complex Dense）や単純な線形射影（Linear Projection）のみに置き換えるアプローチが考えられる。これにより、Baselineとパラメータ数を統一した公平な比較が可能になると同時に、位相情報の直接的な活用により、より軽量かつ高性能なモデルが実現できる可能性がある。
この「位相保存を前提とした特徴抽出部の簡素化」の検証は、本研究の発展的な課題として重要である。


\section{結論}

本研究では、偏波合成開口レーダ（PolSAR）画像の分類において、データの複素数情報を数学的に厳密に扱うことを目的とし、完全な複素数対応アーキテクチャを備えた Vision Transformer モデルを提案した。
従来の Split-Complex アプローチが抱えていた位相情報の損失や歪みの問題を解決するため、Complex Multi-Head Attention、Complex Layer Normalization、および位相保存型活性化関数（ModReLU等）を導入し、理論的な整合性を担保した。

3つの異なるデータセット（Flevoland, San Francisco, Baltrum）を用いた評価実験の結果、提案手法はFlevolandおよびSan Franciscoデータセットにおいて、位相情報を破棄するBaseline手法と同等の高い分類精度（OA 98\%前後）を達成し、その有効性を示した。特に「Water」クラスのような特定の地物においては、Baselineを凌駕する識別能力を確認した。
一方で、より複雑なBaltrumデータセットにおいては、Baselineに比べて若干の精度低下が見られた。考察で述べたように、これは完全な複素演算の導入に伴う実質的なパラメータ数の減少や、ModReLUによるスパース性の欠如が、モデルの学習容量や表現力を制限したことに起因すると示唆される。

今後の展望として、位相保存の利点を維持しつつモデルの表現力を回復させるための、新たな複素活性化関数の設計やバイアス初期化の改善が挙げられる。また、位相情報の伝播が保証された環境下における、初期特徴抽出部（CV-CNN）の簡素化・軽量化も有望な方向性である。
本研究は、PolSARデータ解析におけるDeep Learning、特に複素数ニューラルネットワーク（CVNN）の可能性と課題を明らかにするものであり、今後のより物理的整合性の高いモデル開発への重要な指針を与えるものである。




```