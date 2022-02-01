# 開発者用 README

## 開発手順

### 仮想環境の作成
本パッケージの開発は `venv` という仮想環境上で行うものとする．
`/bin` は仮想環境周りのバイナリが揃っているフォルダである．
環境に入るコマンドは `source bin/activate`，環境が出るコマンドは `deactivate` となる．
python のバージョンが変わっていたり，`pip list` が空(環境下で install したもののみ)になっていたら環境が切り替わっている．
また，`/requirements.txt` に開発に必要なパッケージをリストアップしているので，`pip install -r requirements.txt` を実行すれば記載されているパッケージを全て install できる．

### 開発方法について
`src/si4ul` 内のファイルを編集しながら開発するのはとても効率が悪いので，`/notebook` に開発用の notebook を作成して，そこで行うようにするべき．
ただ，パッケージ内に実装されている既存の関数やクラスを使いたい場合は import して使った方が楽ではあるが，その関数やクラスを編集する必要があるならば，notebook にコピペした方が良い．notebook で関数もしくはクラス形式で実装が完了したら， `src/si4ul` に**設計に基づいて**ファイルやソースコードを追加し，開発版パッケージを通して検証を行ってほしい．

### ソースコードの設計について
`/src/si4ul` に，API ファイルと以下のドメインフォルダを作成してある．
- `experiments`
- `plot`
- `si`

フロー(依存関係)は次のようになる．

API -> `experiments` -> `plot`/`si`

API と `experiments` は関数単位で 1:1 対応させておき，`experiments` は任意の `plot` および `si` のクラスや関数を呼び出すことができる．
イメージがつかない場合は，実際にソースコードを見れば理解が深まるだろう．

それぞれの役割について説明する．

#### API ファイル
ここでは，パッケージで実際に呼び出される関数を宣言する．
##### ここで実装すべき内容
- experiments モジュールの呼び出し
    - `experimets` 内に同名ファイルを作り，同名関数を用意して，そこに処理の本体を記述しておく
- 省略可能引数(引数の初期値)の定義
- 引数に対するエラーハンドリング
- ドキュメント用のコメント(docstring)
##### ここで実装してはいけない内容
- 処理の本体
    - `experiments` に同名ファイルと同名関数を作成してそこに書く
- `plot` および `si` モジュールの import
    - これらを import したいような内容は `experiments` の役割です

#### experiments
ここでは，API で定義された関数の処理の本体を `si` および `plot` の関数を適宜呼び出しながら記述する．
##### ここで実装すべき内容
- API で定義された関数の処理の本体
##### ここで実装してはいけない内容
- 引数に対するエラーハンドリング
    - 呼び出し元の API ファイルの役割です
- plot に関する実装
    - `plot` の役割ですので，`plot` で(なるべく汎用的に)関数やクラスを作成して呼び出してください
- si に関する実装
    - `si` の役割ですので，`si` で(なるべく汎用的に)関数やクラスを作成して呼び出してください

#### plot
ここでは，plot に関する実装を行う．
任意の `experiments` から呼び出される．
##### ここで実装すべき内容
- plot に関する実装
##### ここで実装してはいけない内容
- plot に関する実装以外
- `si` の import
    - `experiments` の役割ですので，実装内容を見直してください

#### si
ここでは，si に関する実装を行う．
任意の `experiments` から呼び出される．
##### ここで実装すべき内容
- si に関する実装
    - sicore を駆使して実装するようにしましょう
- ドキュメント用のコメント(docstring)
##### ここで実装してはいけない内容
- si に関する実装以外
- `plot` の import
    - `experiments` の役割ですので，実装内容を見直してくださ


### ドキュメントの作成について
`src/docs` では，`sphinx` というパッケージを用いて ソースコードないの docstring を基にドキュメントを自動作成するためのファイルが揃っている．
ドキュメントを作成(コンパイル)するコマンドは `make latexpdf` となっており，`si4ul.pdf` が作成される．
ドキュメントを更新する際は，上記コマンドで得られた PDF をホーム直下に移動(置き換え)することで完了する．
pypi の si4ul のページには，github 上にあるこのPDFを参照するようにURLを指定してある(ホーム直下にないと参照できない)


## 検証手順

### 開発版のインストール
開発版の `si4ul` パッケージを使う(install)するには，`pip install -e [si4ulのルートパス]` を実行する．
`pip list` で `si4ul    [バージョン]    [si4ulのルートパス]` が確認できていれば成功．
もし公開されている si4ul を install している場合は uninstall する必要がある．

### ノートブックでの実行
`/notebook` に検証用の `.ipynb` が入っている．
ソースコードの変更を反映させるには，restart を実行してから再度 `import si4ul` を実行する必要がある．

### テストについて
余力があればやりますが，期待しないでください．