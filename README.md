## ダウンロード

研究するPC(linux)上で以下を実行

```bash
git clone https://github.com/spark856/consensus_build_using_RL
```


## 動かすために必要なこと

pipenvがうまくいかないようなので、pip3で入れましょう。

1. 次ページのライブラリを一通りいれる  
    https://qiita.com/inoory/items/e63ade6f21766c7c2393

2. tensorflow,matpoltlibの導入  



## ディレクトリ・ファイル説明

data: 
  - 実験結果の出力先

demands:
  - ユーザパターンの保存先

jupyter:
  - jupyter note book で開くファイル　主にplotで使う
  
myenv:
  - 自作環境の保存先
  
outside:
  - 外気温データ

play:
  - 学習、テストなどの実行ファイル

sh:
  - 一括処理のためのスクリプトファイル

DQN.py:
  - DQNを使うためのプログラム

ddpg.py:
  - ddpgを使うためのプログラム(うまくいかなかった)

temp_pre.py:
  - 外気温を気象庁ホームページからとってくるプログラム

