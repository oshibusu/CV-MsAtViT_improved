# Baltrum データセット対応の実装計画

`CV-MsAtViT_default_donoteditcode` 内のコードを修正して、`Datasets` ディレクトリ内の Baltrum データセットを使用できるようにするための手順です。

## 1. Load_Data.py の修正

Baltrumのデータを読み込むロジックを追加する必要があります。

**変更内容:**
`load_data` 関数内に、`Baltrum` で始まるデータセット名が渡された場合の処理を追加します。

```python
    elif name.startswith('Baltrum'):
        # 例: Baltrum_S_FP1
        _, band, fp = name.split('_')
        # パスは環境に合わせて調整してください。以下は相対パスの例です。
        # ../Datasets/... となっているのは、現在のディレクトリの親にあるDatasetsを参照する場合です。
        # 実行ディレクトリ直下にDatasetsがある場合は ../ を削除してください。
        base_path = Path(f'Datasets/Baltrum/dataset/Pol-InSAR-Island_updated/data/{fp}/{band}/T6')
        label_base_path = Path(f'Datasets/Baltrum/dataset/Pol-InSAR-Island_updated/label/{fp}')

        # T6データから必要な成分を読み込んで T3 形式 (6チャンネル) に整形
        first_read = envi.open(base_path / 'T11.bin.hdr', base_path / 'T11.bin').read_band(0)
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)

        T[:, :, 0] = first_read # T11 (Real)
        T[:, :, 1] = envi.open(base_path / 'T22.bin.hdr', base_path / 'T22.bin').read_band(0) # T22 (Real)
        T[:, :, 2] = envi.open(base_path / 'T33.bin.hdr', base_path / 'T33.bin').read_band(0) # T33 (Real)
        
        # Off-diagonal elements (Complex)
        T[:, :, 3] = envi.open(base_path / 'T12_real.bin.hdr', base_path / 'T12_real.bin').read_band(0) + \
                     1j * envi.open(base_path / 'T12_imag.bin.hdr', base_path / 'T12_imag.bin').read_band(0)
        T[:, :, 4] = envi.open(base_path / 'T13_real.bin.hdr', base_path / 'T13_real.bin').read_band(0) + \
                     1j * envi.open(base_path / 'T13_imag.bin.hdr', base_path / 'T13_imag.bin').read_band(0)
        T[:, :, 5] = envi.open(base_path / 'T23_real.bin.hdr', base_path / 'T23_real.bin').read_band(0) + \
                     1j * envi.open(base_path / 'T23_imag.bin.hdr', base_path / 'T23_imag.bin').read_band(0)

        # ラベルの読み込みと結合 (Train + Test)
        label_train = envi.open(label_base_path / 'label_train.bin.hdr', label_base_path / 'label_train.bin').read_band(0)
        label_test = envi.open(label_base_path / 'label_test.bin.hdr', label_base_path / 'label_test.bin').read_band(0)
        labels = label_train + label_test
```

## 2. SAR_utils.py の修正

Baltrumデータセットのクラス名定義と、クラス数を返すロジックを追加します。

**変更内容:**

`target(name)` 関数に追加:
```python
    elif 'Baltrum' in name:
        target_names = ['Unassigned', 'Tidal flat', 'Water', 'Coastal shrub', 'Dense, high vegetation', 'White dune', 
                        'Peat bog', 'Grey dune', 'Couch grass', 'Upper saltmarsh', 'Lower saltmarsh', 'Sand', 'Settlement']
```

`num_classes(dataset)` 関数に追加:
```python
    elif 'Baltrum' in dataset:
        output_units = 12
```

## 3. main_CV_MsAtViT.py の修正

使用するデータセット名を変更します。

**変更内容:**

```python
# Get the data
dataset = 'Baltrum_S_FP1' # ここを変更 (例: Baltrum_S_FP1, Baltrum_L_FP2 など)
```

また、Baltrumは画像サイズが大きい場合があるため、メモリ不足になる場合は `createImageCubes` の引数でパッチ生成数を制限したり、`dataset` の条件分岐でラーニングレートを調整するロジック（`ober` と同様）を追加する必要があるかもしれません。
