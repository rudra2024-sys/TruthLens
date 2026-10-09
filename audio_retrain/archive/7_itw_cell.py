import subprocess, zipfile

ITW_DIR = f'{DATA_DIR}/itw_subset'
ITW_BACKUP = f'{PROJECT_DIR}/itw_subset_backup.tar.gz'
ITW_MANIFEST = f'{MANIFEST_DIR_V7}/itw_manifest.csv'
ITW_MIRRORS = ['abdallamohamed312/in-the-wild-audio-deepfake',   # real/ + fake/ folders + meta.csv
               'abdallamohamed312/in-the-wild-dataset']          # backup copy (original release layout)
ITW_MIN_FILES = 30000     # the full dataset has 31,779 clips; fewer than this means a partial download
NESTED_MIN_BYTES = 1e9    # a zip bigger than this inside the download is treated as the real data archive

# caps per class per split (total ~8k clips)
ITW_CAP = {'train': 2500, 'val': 400, 'test': 800}
MAX_CLIP_SECONDS = 12

def kaggle_credentials_status():
    has_env = bool(os.environ.get('KAGGLE_API_TOKEN'))
    has_file = os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))
    print(f'Kaggle credentials in this session -> env token: {has_env}, kaggle.json: {has_file}')
    if not (has_env or has_file):
        print('  NO Kaggle credentials found. Run the "Kaggle credentials" cell near the top first.')

def list_dir(raw_dir):
    for name in sorted(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, name)
        size = os.path.getsize(p) if os.path.isfile(p) else 0
        print(f'    {name}  {"(dir)" if os.path.isdir(p) else f"{size / 1e9:.2f} GB"}')

def extract_nested_archives(raw_dir):
    # This Kaggle mirror ships the data as a zip INSIDE the downloaded archive (a file literally named
    # 'download'), so --unzip leaves a big zip behind. Unpack it (no re-download needed).
    if not os.path.isdir(raw_dir):
        return
    for _ in range(3):
        if len(glob.glob(f'{raw_dir}/**/*.wav', recursive=True)) >= ITW_MIN_FILES:
            return
        found = False
        for p in glob.glob(f'{raw_dir}/**/*', recursive=True):
            if os.path.isfile(p) and os.path.getsize(p) > NESTED_MIN_BYTES and zipfile.is_zipfile(p):
                print(f'  Unpacking nested archive {p} ({os.path.getsize(p) / 1e9:.2f} GB) ...')
                subprocess.run(['unzip', '-q', '-o', p, '-d', f'{raw_dir}/extracted'])
                os.remove(p)   # free the disk space; the extracted copy is what we use
                found = True
        if not found:
            return

def download_itw(raw_dir):
    kaggle_credentials_status()
    for ref in ITW_MIRRORS:
        shutil.rmtree(raw_dir, ignore_errors=True)   # never build on top of a partial download
        os.makedirs(raw_dir, exist_ok=True)
        print(f'Downloading {ref} (~9 GB, roughly 5-15 min, no progress bar) ...')
        r = subprocess.run(['kaggle', 'datasets', 'download', '-d', ref, '-p', raw_dir, '--unzip'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            extract_nested_archives(raw_dir)
        n = len(glob.glob(f'{raw_dir}/**/*.wav', recursive=True))
        if r.returncode == 0 and n >= ITW_MIN_FILES:
            print(f'  OK: {n} wav files')
            return ref
        print(f'  FAILED (exit code {r.returncode}, {n} wav files found). Kaggle said:')
        print('  ' + ((r.stderr or '') + (r.stdout or ''))[-800:].replace('\n', '\n  '))
        print('  Files left in the download folder:')
        list_dir(raw_dir)
    raise RuntimeError('Could not download In-The-Wild from any mirror -- read the Kaggle messages above.')

def meta_label_to_class(x):
    x = str(x).lower()
    if x in ('bona-fide', 'bonafide', 'bona_fide', 'real', 'genuine'):
        return 'genuine'
    if x in ('spoof', 'fake', 'deepfake'):
        return 'deepfake'
    return None

def restore_itw():
    subprocess.run(['cp', ITW_BACKUP, '/content/itw_subset_backup.tar.gz'], check=True)
    subprocess.run(['tar', '-xzf', '/content/itw_subset_backup.tar.gz', '-C', DATA_DIR], check=True)
    return pd.read_csv(ITW_MANIFEST)

def build_itw():
    raw_dir = f'{DATA_DIR}/itw_raw'
    extract_nested_archives(raw_dir)   # salvage a download already on disk instead of fetching 9 GB again
    if len(glob.glob(f'{raw_dir}/**/*.wav', recursive=True)) < ITW_MIN_FILES:
        download_itw(raw_dir)

    all_wavs = glob.glob(f'{raw_dir}/**/*.wav', recursive=True)
    meta_path = glob.glob(f'{raw_dir}/**/meta.csv', recursive=True)
    assert meta_path, 'meta.csv (speaker info) not found'
    meta = pd.read_csv(meta_path[0])
    print('meta.csv columns:', meta.columns.tolist())
    print(meta.head(3).to_string())
    file_col = next(c for c in meta.columns if 'file' in c.lower())
    spk_col = next(c for c in meta.columns if 'speaker' in c.lower())
    label_col = next((c for c in meta.columns if 'label' in c.lower()), None)
    meta['_name'] = meta[file_col].astype(str).map(lambda x: Path(x).name if x.lower().endswith('.wav') else x + '.wav')
    speaker_of = dict(zip(meta['_name'], meta[spk_col].astype(str)))

    # label: from the real/ fake/ folders if present, otherwise from meta.csv (original release layout)
    folder_label = {}
    for p in all_wavs:
        parent = Path(p).parent.name.lower()
        if parent in ('real', 'fake'):
            folder_label[p] = 'genuine' if parent == 'real' else 'deepfake'
    if len(folder_label) < 0.9 * len(all_wavs):
        assert label_col is not None, 'No real/fake folders and no label column in meta.csv'
        name_to_label = dict(zip(meta['_name'], meta[label_col].map(meta_label_to_class)))
        folder_label = {p: name_to_label.get(Path(p).name) for p in all_wavs}
    rows = []
    for p, label in folder_label.items():
        spk = speaker_of.get(Path(p).name)
        if label is not None and spk is not None:
            rows.append({'path': p, 'label': label, 'speaker_id': spk})
    df = pd.DataFrame(rows)
    print(f'{len(df)} clips matched to a speaker and a label; classes:', df['label'].value_counts().to_dict())
    assert df['label'].nunique() == 2, 'Need both real and fake clips'
    if label_col is not None:  # sanity: our label vs meta's label agree
        name_to_meta = dict(zip(meta['_name'], meta[label_col].astype(str)))
        df['_meta_label'] = df['path'].map(lambda p: name_to_meta.get(Path(p).name))
        print(pd.crosstab(df['label'], df['_meta_label']))

    # split BY SPEAKER; both classes of a person stay in the same split
    speakers = sorted(df['speaker_id'].unique())
    rs = np.random.RandomState(42)
    rs.shuffle(speakers)
    n_test, n_val = max(1, round(0.2 * len(speakers))), max(1, round(0.1 * len(speakers)))
    spk_split = {}
    for i, s in enumerate(speakers):
        spk_split[s] = 'test' if i < n_test else ('val' if i < n_test + n_val else 'train')
    df['split'] = df['speaker_id'].map(spk_split)
    print(f'{len(speakers)} speakers -> train {sum(v == "train" for v in spk_split.values())}, '
          f'val {n_val}, test {n_test}')

    # subset: equal real/fake per split, capped
    parts = []
    for split, cap in ITW_CAP.items():
        d = df[df['split'] == split]
        n = min((d['label'] == 'genuine').sum(), (d['label'] == 'deepfake').sum(), cap)
        for label in LABELS:
            parts.append(d[d['label'] == label].sample(n, random_state=42))
    sub = pd.concat(parts).reset_index(drop=True)

    # convert to 16 kHz mono FLAC (compact + one uniform container for every clip)
    new_paths = []
    for i, r in tqdm(sub.iterrows(), total=len(sub), desc='converting to flac'):
        out_dir = f'{ITW_DIR}/{r["split"]}/{r["label"]}'
        os.makedirs(out_dir, exist_ok=True)
        out = f'{out_dir}/{i}.flac'
        if not os.path.exists(out):
            audio = load_mono_16k(r['path'])[:MAX_CLIP_SECONDS * TARGET_SR]
            sf.write(out, audio, TARGET_SR, format='FLAC')
        new_paths.append(out)
    sub['path'] = new_paths
    sub['source'] = 'inthewild'
    sub = sub[['path', 'label', 'source', 'speaker_id', 'split']]
    sub.to_csv(ITW_MANIFEST, index=False)

    subprocess.run(['tar', '-czf', '/content/itw_subset_backup.tar.gz', '-C', DATA_DIR, 'itw_subset'], check=True)
    size = os.path.getsize('/content/itw_subset_backup.tar.gz')
    free = shutil.disk_usage(PROJECT_DIR).free
    if size * 1.05 < free:
        shutil.copy2('/content/itw_subset_backup.tar.gz', ITW_BACKUP)
        ok = os.path.exists(ITW_BACKUP) and os.path.getsize(ITW_BACKUP) == size
        print(f'ITW backup to Drive: {size / 1e9:.2f} GB, saved={ok}')
    else:
        print(f'WARNING: Drive too full for the {size / 1e9:.2f} GB ITW backup ({free / 1e9:.2f} GB free) -- '
              'skipped. Next session will re-download In-The-Wild.')
    shutil.rmtree(raw_dir, ignore_errors=True)
    return sub

if os.path.exists(ITW_BACKUP) and os.path.exists(ITW_MANIFEST):
    print('Restoring In-The-Wild subset from Drive backup...')
    itw_df = restore_itw()
else:
    itw_df = build_itw()

print(itw_df.groupby(['split', 'label']).size().unstack())
print('speakers per split:', itw_df.groupby('split')['speaker_id'].nunique().to_dict())
train_spk = set(itw_df[itw_df.split == 'train'].speaker_id)
test_spk = set(itw_df[itw_df.split == 'test'].speaker_id)
assert not (train_spk & test_spk), 'speaker leak between train and test!'
