# Google Colab LLM Servisleri

Büyük model servislerini Google Colab GPU'larında çalıştırır ve Cloudflare Tunnel (varsayılan) ya da FRP + Caddy üzerinden internete açar.
**"Bir servis = bir notebook (`src/colabllm/services/<name>.ipynb`) + bir çalıştırma betiği (`src/colabllm/services/<name>.py`)"** kuralına uyar.

> Bu depo [deluxebear/colabllm](https://github.com/deluxebear/colabllm) projesinin çevrilmiş bir kopyasıdır: kod, CLI çıktıları, arayüz metinleri ve belgeler İngilizcedir; bu README Türkçedir.

## Tek komut girişi

Kullanıcılar dağıtım paketiyle kurar, kaynak kodu klonlamaya gerek yoktur. Wheel dosyasını [GitHub Releases](https://github.com/deluxebear/colabllm/releases) sayfasından indirip kurun (özel depo ise erişim izni gerekir):

```bash
uv tool install ./colabllm-0.1.4-py3-none-any.whl
# Komut dizini henüz PATH'te değilse
uv tool update-shell
```

Sürümler PyPI'a yayınlandıktan sonra doğrudan oradan da kurulup güncellenebilir:

```bash
uv tool install colabllm
uv tool upgrade colabllm
```

Ardından herhangi bir dizinde `colabllm <runtime-adı> <komut> [argümanlar...]` kullanılabilir:

```bash
colabllm services                       # tüm runtime'ları listeler
colabllm unlimited-ocr up                # varsayılan A100
colabllm unlimited-ocr ocr ./documents -O ./results
colabllm unlimited-ocr resume JOB_ID --wait
colabllm qwen3-tts up -i 2
colabllm comfyui up
colabllm ollama up --dry-run
colabllm unlimited-ocr --help

# Proje geneli yapılandırma da aynı girişi kullanır
colabllm tunnel-config
colabllm frp-config --help
colabllm frp-status
```

Tüm çalıştırma betikleri ve notebook'lar kurulum paketine dahildir. Yerel göreli girdi/çıktı yolları komutun çalıştırıldığı dizine göre çözülür.
Projenin yapılandırması ve çalışma durumu `~/.colabllm/` altına yazılır; depodaki eski `.state/` okunmaz ve eski yapılandırma taşınmaz.
Ayrı bir dizin kullanmak için `COLABLLM_HOME=/absolute/path` ayarlanabilir.

```text
~/.colabllm/
├── tunnel-config.json           # global ana alan adı ve FRP ayarları
├── services/<name>/state.json   # örnek kimlik bilgileri, oturum, tünel ve runner PID
├── services/<name>/run.log      # log; çalıştırılan notebook da örnek dizinindedir
└── frp/<server-id>.json         # FRP gizli kimlik bilgileri
```

`colab`, `cf` ve `ssh` için yetkilendirme/yapılandırma kendi araçlarında yönetilir; Google Drive verileri uzakta kalır.
Kaldırmak için `uv tool uninstall colabllm`; kullanıcı yapılandırma dizini korunur.
Güncellemeden önce çalışan servisleri durdurun, sonra yeniden başlatın; böylece arka plandaki runner'ın kullandığı kurulu dosyalar değiştirilmez.

## Runtime kullanım kılavuzları

Her runtime'ın komutları, API/arayüz akışları, girdi/çıktıları ve kurtarma davranışı aşağıdaki "Runtime kılavuzları" tablosundaki belgelerdedir.
Ortak kurallar [Ortak runtime işlemleri](https://github.com/deluxebear/colabllm/blob/master/docs/runtime-common.md) belgesinde toplanmıştır; kimlik doğrulama, Drive yolları, zaman aşımları ve çoklu örnekleri anlamak için önce onu okuyun.
Kullanım belgelerinin kaynağı `docs/<runtime>.md` dosyalarıdır; kullanıcı skill'indeki karşılıkları senkronizasyon aracıyla üretilir:

```bash
python tools/sync_runtime_docs.py
python tools/sync_runtime_docs.py --check
```

Kullanım değiştiğinde ya da yeni runtime eklendiğinde senkronize edin; yerel testler ve CI, kılavuzların eksiksizliğini, örnek argümanları ve skill içeriğinin tutarlılığını denetler.

## Agent Skill kurulumu

Kökteki [skills/colabllm](https://github.com/deluxebear/colabllm/blob/master/skills/colabllm/SKILL.md), bu projenin tüm runtime'larını kapsayan, kullanıcıya yönelik bağımsız bir skill sunar.
Görev bazında yüklenen OCR, ses, çıkarım, medya ve sorun giderme referansları içerir; kaynak depoya veya kişisel kurulum ayarlarına bağlı değildir.

Wheel ve kaynak paketi skill'in tamamını içerir; CLI kurulduktan sonra dışa aktarılabilir:

```bash
# Ana skill açıklamasını göster
colabllm skill

# Codex: colabllm/ skill dizininin tamamını kişisel skill dizinine aktar
colabllm skill --output "${CODEX_HOME:-$HOME/.codex}/skills"

# ~/.agents/skills destekleyen agent'lar
colabllm skill --output ~/.agents/skills

# Diğer agent'lar: gerçek skill dizinini belirtin
colabllm skill --output /path/to/agent/skills
```

`--output` skill'lerin üst dizinidir; asıl olarak `<dizin>/colabllm/SKILL.md` ve `references/` oluşturulur.
Hedef zaten varsa hata verir, kullanıcı değişikliklerinin üzerine yazmaz. Güncellemek için önce yeni bir dizine aktarın, karşılaştırıp değiştirin; CLI güncellemesi kurulu skill'in üzerine otomatik yazmaz.
`skills/colabllm/` dizininin tamamı kaynaktan doğrudan da kopyalanabilir. Kurduktan sonra agent'ınızın yöntemiyle skill'leri yeniden yükleyin ya da yeni bir oturum açın,
ardından "colabllm ile yerel PDF'i Markdown'a çevir" gibi görevler isteyin.

Depodaki `.agents/skills/colab-llm-service/` bakımcı skill'idir; kullanıcılara dağıtılan `skills/colabllm/`'dir.

## Geliştirme, derleme ve yayınlama

```bash
# Geliştirme kurulumu: kaynak değişiklikleri hemen geçerli olur, kaynak dizini yerinde kalmalı
uv tool install --editable .

# Normal yerel kurulum / güncelleme
uv tool install --force .

# Wheel ve kaynak dağıtım paketini derle (dist/)
uv build --no-sources

# Wheel'in izole ortamda kurulup çalıştığını doğrula
python tools/check_distribution.py dist/colabllm-0.1.4-py3-none-any.whl
```

Sürüm numarasının tek kaynağı `pyproject.toml`'dur; CLI sürümü kurulu paketin meta verisinden okur.
`MAJOR.MINOR.PATCH` kullanılır: düzeltmeler PATCH'i, yeni özellikler MINOR'u, uyumsuz değişiklikler MAJOR'u artırır;
`0.x` geliştirme aşamasında uyumsuz değişiklikler MINOR'u artırır ve [CHANGELOG.md](https://github.com/deluxebear/colabllm/blob/master/CHANGELOG.md) içinde açıklanır.
Güncel sürüm `0.1.4`, Git etiketi `v0.1.4`'tür (`0.1.0` geri çekildi); sonraki bir yayın örneği:

```bash
uv version --bump patch
# CHANGELOG.md içindeki Unreleased bölümünü bu sürümün tarihli girdisine taşı
python tools/check_release.py --tag v0.1.5
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release 0.1.5"
git tag -a v0.1.5 -m "Release 0.1.5"
git push origin master
git push origin v0.1.5
```

`v*` etiketinin gönderilmesi [release.yml](https://github.com/deluxebear/colabllm/blob/master/.github/workflows/release.yml) iş akışını tetikler:

1. Etiket, paket sürümü ve değişiklik kaydının tutarlılığını doğrular; Linux/macOS × Python 3.12–3.14 CI'ını yeniden kullanır.
2. Wheel ve sdist'i derler, izole kurulumu doğrular, SHA256SUMS üretir ve Actions artifact olarak saklar.
3. Kurulum paketleri ve sağlama dosyalarıyla bir GitHub Release yayınlar; sürüm notları güncel sürümün CHANGELOG girdisinden alınır.
4. Doğrulanmış aynı paketleri Trusted Publishing ile PyPI'a yükler.

PyPI'ı ilk kez etkinleştirmek için proje sahibinin [PyPI Publishing](https://pypi.org/manage/account/publishing/) sayfasında
bir Pending Publisher eklemesi gerekir: proje `colabllm`, Owner `deluxebear`, Repository `colabllm`,
Workflow `release.yml`, Environment `pypi`. PyPI projesi zaten varsa aynı yayıncı projenin Publishing ayarlarına eklenir.
Actions, `pypi` environment ve OIDC kullanır; uzun ömürlü bir PyPI API token'ı gerekmez.
Bkz. [PyPI Trusted Publisher açıklaması](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

PyPI yüklemesi başarısız olsa da GitHub Release indirilebilir; kimlik ayarını düzelttikten sonra Actions'ta **Re-run failed jobs** seçin.
Mevcut bir etiket için release iş akışı elle de çalıştırılabilir; yayınlanmış bir GitHub Release'in üzerine yazılamaz, bu yüzden önce yalnızca başarısız işleri yeniden çalıştırın.
Yayınlanmış sürümlerde etiket taşınmaz ve paketler değiştirilmez; düzeltmeden sonra yeni sürüm yayınlanır.
Yerel wheel ile kuranlar yeni wheel'i kurarak güncelleyebilir; PyPI ile kuranlar `uv tool upgrade colabllm` kullanır.

## CLI davranış kuralları

- `colabllm <servis> --help` yaşam döngüsü ve servise özel komutları listeler; ayrıntılı yardım için `colabllm help <servis> <komut>`.
- `-i/--instance` servis argümanlarının herhangi bir yerine yazılabilir, `--` sonrası olduğu gibi yorumlanır; örnek kimliği 1–64 karakterlik harf, rakam, alt çizgi veya tireden oluşmalı ve tireyle başlamamalıdır.
- Normal sonuçlar stdout'a, ilerleme ve uyarılar stderr'e yazılır; `--json` ile stdout yalnızca JSON içerir.
- `info`, `ps`, `list`, `status` `--json` destekler ve kimlik bilgilerini varsayılan olarak gizler; gerektiğinde açıkça `--show-secrets` ekleyin.
- `--quiet` tanılama ilerlemesini bastırır, hatalar yine gösterilir; `--no-color` veya `NO_COLOR` tanılama renklerini kapatır. Terminal olmayan çıktıda renkler otomatik kapanır.
- Yaygın ağ, dosya ve yapılandırma hataları varsayılan olarak kısa ipuçlarıyla gösterilir; `--debug` istisna yığınını gösterir.
- Başarıda çıkış kodu 0; işlem hatasında sıfırdan farklı, argüman kullanım hatasında 2; dış Colab komutu başarısız olursa onun pozitif çıkış kodu korunur; Ctrl+C 130.
- Hazır işareti olarak yalnızca tam bir `CT_READY` protokol satırı kabul edilir; loglarda geçen sıradan metinler hazır sayılmaz.
- `up` varsayılan olarak `CT_READY` bekler; bekleme zaman aşımına uğrarsa hata döner ama arka plandaki runner kalır. `up --detach` yalnızca arka plan runner'ını başlatıp hemen döner; `up --foreground` bağlantıyı mevcut terminalde tutar.
- Aynı örnek için başlatma ve durdurma birbirini dışlar; durum işlemleri dosya kilidi ve atomik değiştirme kullanır. Ön plan/arka plan runner'ları süreç başlangıç kimliğini kaydeder, durdurmadan önce kimliği doğrular ve çıkışı bekler.
- `down` önce uzaktaki durdurmanın başarılı olduğunu doğrular, sonra yerel bağlantıyı durdurup çalışma durumunu temizler; uzaktaki durdurma başarısız olursa yeniden deneme için bağlantı ve takip bilgileri korunur.
- Kimlik bilgisi taşıyan HTTP istekleri farklı kökene yönlendirmeyi reddeder. `--url` veya servis URL ortam değişkeniyle köken değiştirildiğinde token'ın açıkça verilmesi gerekir; asıl servisin kimlik bilgileri otomatik ödünç alınmaz. Köken; şema, ana makine ve geçerli porta göre karşılaştırılır.

## Web kontrol paneli

```bash
colabllm ui                 # tarayıcıda açar; varsayılan port 8765
colabllm ui --port 0 --no-browser
```

Yalnızca `127.0.0.1` üzerinde dinler ve her başlatmada yeni bir erişim token'ı üretir; terminalde yazdırılan bağlantıyı kullanın.
Panel; compute unit bakiyesi ve harcama hızı, tüm örneklerin durumu/GPU'su/URL'si/çalışma süresi, servis kataloğu, `doctor` denetimleri, `run.log` ve gizlenmiş kimlik bilgilerini gösterir.
Başlatma ve durdurma CLI ile aynı servis betiklerini (`up --detach`, `down`) çalıştırır; kilitler ve hazır olma denetimleri değişmez.
Başlatma penceresi her servisin kendi `up` seçeneklerini (ör. comfyui `--workflow`, ollama `--models`) CLI ayrıştırıcısından okur ve göndermeden önce aynı ayrıştırıcıyla doğrular.
Örnekler listesinde birden fazla örnek seçilip birlikte, ya da tüm çalışan örnekler tek seferde durdurulabilir; her biri ayrı bir `down` işlemi olarak izlenir.
Durmuş örnekler panelden (tek tek ya da seçerek) silinebilir; bu, `colabllm <runtime> purge` komutunu çalıştırır: önce `colab stop` ile arta kalan oturum kapatılır, sonra yerel kayıt (kimlik bilgileri, `run.log`, notebook kopyası) silinir. Drive verilerine dokunulmaz.
Paneli kapatmak (Ctrl+C) çalışan servisleri etkilemez.

## Zaman aşımları ve ortam tanılama

```bash
# Dış komut üst sınırı: servis adından önce veya sonra yazılabilir, COLABLLM_COMMAND_TIMEOUT ile de ayarlanabilir
colabllm --command-timeout 60 ollama status

# Varsayılan modda zaman aşımı yalnızca beklemeyi durdurur, arka plan runner'ı kalır; ön plan modunda yerel bağlantı kesilir
colabllm ollama up --ready-timeout 600
colabllm ollama up --foreground --ready-timeout 600

# Yoklama bekleme sınırı: 0 sınırsız bekler; her istek de kalan süreyle sınırlanır
colabllm unlimited-ocr job JOB_ID --wait --timeout 300
colabllm comfyui pull-workflow TEMPLATE --timeout 300 --poll 2
colabllm longcat-avatar job JOB_ID --wait --timeout 300

# Varsayılan olarak yalnızca yerel bağımlılıkları, Drive eklentisini, yetki dosyalarının varlığını ve kullanıcı yapılandırmasını denetler
colabllm doctor
colabllm doctor --json --transport quick --no-drive

# Ek olarak Colab, Cloudflare ve SSH yetkilerini seçilen arka uca göre süre sınırlı, salt okunur şekilde doğrular
colabllm doctor --online --timeout 10
```

Zaman aşımı çıkış kodu `124`'tür. Uzun süreli `colab exec` canlı tutma bağlantısı genel dış komut sınırından etkilenmez.
`--command-timeout` verilmezse ortak dış komutlar için varsayılan 300 saniyedir; bazı indirme veya SSH işlemlerinin kendi varsayılan sınırları vardır.
`doctor` GPU oturumu oluşturmaz, Drive bağlamaz, tünelleri değiştirmez ve kimlik bilgilerini ya da dış komutların ham çıktısını göstermez.
Çevrimdışı denetim yetkilerin hâlâ geçerli olduğunu kanıtlamaz; gerektiğinde `--online` kullanın, bu mod tarayıcı etkileşimini kapatır ve her denetimi süreyle sınırlar.

## Shell tamamlama

Komutlar, seçenekler ve sabit değerler gerçek ayrıştırıcıdan gelir; örnek kimlikleri `~/.colabllm/services/` altındaki dizin adlarından okunur, tamamlama sırasında kimlik bilgisi okunmaz ve ağa çıkılmaz.

```bash
# Bash: mevcut shell'e yükle
source <(colabllm completion bash)

# Zsh: önce tamamlama sistemini etkinleştirin, sonra yükleyin
# autoload -Uz compinit && compinit
source <(colabllm completion zsh)
```

Fish'te `colabllm completion fish | source` çalıştırın. Kalıcı etkinleştirmek için ilgili yükleme komutunu kendi shell başlangıç dosyanıza ekleyin
ya da betiği `~/.colabllm/completions/` altına kaydedip başlangıç dosyasından yükleyin.

## Testler

Yerel CLI ve kontrol mantığı yalnızca Python standart kütüphanesini kullanır; varsayılan testler Torch, NumPy veya GPU gerektirmez:

```bash
python -m unittest discover -s tests
```

`tests_colab/`, Colab tarafındaki model kodunun regresyon testlerini ayrıca saklar (ör. LongCat attention ve ses bölütleme).
Bunlar model çalışma kodunu içe aktarır ve yalnızca ilgili Colab servis ortamında çalışır; attention testleri ayrıca CUDA ve upstream LongCat kaynağını gerektirir.
O ortamda proje kaynağını VM'e koyduktan sonra şunu çalıştırın:

```bash
# LongCat-Video VM'i /content/longcat-video/repo, Avatar VM'i /content/longcat-avatar/repo kullanır
PYTHONPATH=/content/longcat-video/repo python -m unittest discover -s tests_colab
```

GitHub Actions; Linux/macOS ve Python 3.12/3.13/3.14 üzerinde yerel testleri, üç shell tamamlama testini, derlemeyi ve izole wheel kurulum doğrulamasını çalıştırır. İş akışı model testi çalıştırmaz, Colab başlatmaz ve paketleri otomatik yayınlamaz.

Yerelde Zsh/Fish yoksa ilgili shell entegrasyon testleri atlanır; CI bu shell'leri kurar.
Yerel testlerdeki gerçek FRP entegrasyon testi için ayrıca `COLABSVC_FRP_TEST_BIN` verilmelidir, aksi halde atlanır.

## Mimari ve sorumluluklar

| Aşama / sorumluluk | Çalıştığı yer | Açıklama |
| --- | --- | --- |
| Colab oturumu oluşturma/yeniden kullanma (`colab new`) | Yerel betik `src/colabllm/services/<name>.py` | Oturum adı servis adıyla aynıdır |
| Tarayıcısız Drive bağlama (`colab dfs mount`) | Yerel betik `src/colabllm/services/<name>.py` | `/content/drive`'a bağlar; notebook'taki `drive.mount()` otomatik atlanır |
| Cloudflare sabit alan adı | Yerel betik (`cf` CLI) | Named tunnel oluşturur/yeniden kullanır, ingress ve DNS CNAME ayarlar; ana alan adı yoksa dinamik URL kullanılır |
| FRP sabit alan adı | Yerel betik (`ssh` + `cf` CLI) | Sunucu tarafında frps/Caddy'yi SSH ile kurar veya yeniden kullanır, Cloudflare DNS ile alan adını sunucu IP'sine yönlendirir |
| Anahtar üretme ve aktarma | Yerel betik -> `--env` | Kullanıcı adı/şifre ve Bearer token'ı otomatik üretir, `~/.colabllm/services/<name>/state.json` (0600) içinde saklar |
| Servisi kurma ve başlatma | Uzak `src/colabllm/services/<name>.ipynb` | Tekrarlanabilir kurulum; servis yalnızca `127.0.0.1` üzerinde dinler |
| Güvenlik katmanı (Caddy) | Uzak `src/colabllm/services/<name>.ipynb` | Kendi kimlik doğrulaması olmayan servislerin (ör. Ollama) önüne Caddy konur: tarayıcı Basic Auth, API `Authorization: Bearer <token>` kullanır |
| `cloudflared` / `frpc` başlatma ve canlı tutma | Uzak `src/colabllm/services/<name>.ipynb` | Son hücre sürekli bloklar ve alt süreçleri izler; yerel arka plan süreci, Colab'ın oturumu geri almaması için `colab exec` bağlantısını açık tutar |

## Runtime kılavuzları

| Servis | Varsayılan GPU | Kimlik doğrulama | Sabit alan adı örneği (ana alan adı `example.com` ise) | Ayrıntılı kılavuz |
| --- | --- | --- | --- | --- |
| **NVIDIA-Nemotron-3.5-Transcribe-30B-A4B** | `G4` | `vLLM` Bearer API Key (`builtin`) | `https://transcribe.example.com` | [docs/nemotron-transcribe.md](https://github.com/deluxebear/colabllm/blob/master/docs/nemotron-transcribe.md) |
| **Qwen3-TTS** | `L4` | Bearer API Key (`builtin`) | `https://qwen3-tts.example.com` | [docs/qwen3-tts.md](https://github.com/deluxebear/colabllm/blob/master/docs/qwen3-tts.md) |
| **IndexTTS-2.5** | `L4` | Bearer API Key (`builtin`) | `https://indextts.example.com` | [docs/indextts.md](https://github.com/deluxebear/colabllm/blob/master/docs/indextts.md) |
| **Unlimited-OCR 2** | `A100` | Caddy (Basic Auth + Bearer) | `https://unlimited-ocr.example.com` | [docs/unlimited-ocr.md](https://github.com/deluxebear/colabllm/blob/master/docs/unlimited-ocr.md) |
| **LongCat-Video-Avatar 1.5** | `G4` | Caddy (Basic Auth + Bearer) | `https://longcat-avatar.example.com` | [docs/longcat-avatar.md](https://github.com/deluxebear/colabllm/blob/master/docs/longcat-avatar.md) |
| **LongCat-Video** | `G4` | Caddy (Basic Auth + Bearer) | `https://longcat-video.example.com` | [docs/longcat-video.md](https://github.com/deluxebear/colabllm/blob/master/docs/longcat-video.md) |
| **Openjutsu (MiniMax H3 Video Studio)** | `G4` | Caddy (Basic Auth + Bearer) | `https://openjutsu.example.com` | [docs/openjutsu.md](https://github.com/deluxebear/colabllm/blob/master/docs/openjutsu.md) |
| **FLUX 3 Image Klein (T2I + Multi-Ref I2I + PUP)** | `G4` | `vLLM-Omni` Bearer API Key (`builtin`) | `https://flux-klein.example.com` | [docs/flux-klein.md](https://github.com/deluxebear/colabllm/blob/master/docs/flux-klein.md) |
| **Qwen-Image-2.1 (T2I + 1-4 Ref RGBA I2I)** | `G4` | `vLLM-Omni` Bearer API Key (`builtin`) | `https://qwen-image.example.com` | [docs/qwen-image.md](https://github.com/deluxebear/colabllm/blob/master/docs/qwen-image.md) |
| **ComfyUI + ComfyUI-Manager** | `L4` | Caddy (Basic Auth + Bearer) | `https://comfyui.example.com` | [docs/comfyui.md](https://github.com/deluxebear/colabllm/blob/master/docs/comfyui.md) |
| **Qwen3.8-Flash-Next (800K + Vision)** | `G4` | `llama-server` Bearer API Key | `https://qwen.example.com` | [docs/qwen-flash.md](https://github.com/deluxebear/colabllm/blob/master/docs/qwen-flash.md) |
| **DeepSeek-V4.1-Flash Coder (Auto Max 1M + Vision)** | `G4` | `llama-server` Bearer API Key | `https://deepseek.example.com` | [docs/deepseek-coder.md](https://github.com/deluxebear/colabllm/blob/master/docs/deepseek-coder.md) |
| **Ollama** | `T4` | Caddy (Basic Auth + Bearer + `x-api-key`) | `https://ollama.example.com` | [docs/ollama.md](https://github.com/deluxebear/colabllm/blob/master/docs/ollama.md) |
| **Unsloth Studio** | `L4` | Yerleşik web girişi (`builtin`) | `https://unsloth-studio.example.com` | [docs/unsloth-studio.md](https://github.com/deluxebear/colabllm/blob/master/docs/unsloth-studio.md) |

## Dizin yapısı

```text
LICENSE                        # MIT açık kaynak lisansı
pyproject.toml                 # paket meta verisi, sürüm, colabllm komut girişi
src/colabllm/
├── cli.py                     # ortak komut yönlendirme
├── control.py                 # global alan adı ve FRP yönetimi
├── utility.py                 # genel yardımcı alt komutlar
├── lib/                       # yerel ortak runner, durum, HTTP/ses araçları ve FRP uygulaması
└── services/                  # <name>.py + <name>.ipynb; paketle birlikte kurulur
templates/                     # notebook üreticisi build_notebooks.py ve notebook'lara gömülen bağımsız modüller
docs/                          # runtime'lar ve FRP/ortak işlemler belgeleri
skills/colabllm/               # paketle dağıtılan kullanıcı Agent Skill'i (references, docs'tan senkronize üretilir)
tools/                         # belge senkronizasyonu, paket doğrulama, kıyaslama gibi geliştirme araçları
tests/                         # yerel testler (Python standart kütüphanesi)
tests_colab/                   # yalnızca Colab model ortamında çalışan regresyon testleri
```

## Yerel ortam ve ön hazırlık

Yerel makine Colab oturumlarını oluşturur, tünelleri yapılandırır, Drive'ı bağlar ve arka plan bağlantısını sürdürür; modeller uzakta çalışır, **yerelde GPU, CUDA, PyTorch veya model ağırlıkları gerekmez**.

### Zorunlu ve isteğe bağlı bağımlılıklar

| Yerel ortam / araç | Ne zaman gerekir | Açıklama |
| --- | --- | --- |
| Linux veya macOS | Zorunlu | Colab CLI'ın şu an desteklediği platformlar; bu projenin arka plan süreç yönetimi de Unix süreç gruplarını kullanır |
| Python 3.12+ | Zorunlu | `src/colabllm/services/*.py` ve `src/colabllm/control.py`'yi çalıştırır, `google-colab-cli`'ın Python gereksinimini karşılar |
| `uv` | Paket kurulumu ve güncelleme | colabllm ve Colab CLI'ı izole kurar; Git yalnızca geliştirme için gerekir |
| `colab` CLI | Zorunlu | PyPI paketi `google-colab-cli`; oturumlar, dosya işlemleri ve notebook çalıştırma için |
| `colab dfs` eklentisi | Varsayılan Drive akışı için zorunlu | [colab-dfs-ext](https://github.com/deluxebear/colab-dfs-ext); `consent`, `mount`, `doctor` sağlar; ayrıca kurulmalıdır |
| Node.js, npm, Cloudflare `cf` CLI | Sabit alan adı veya FRP modu | Yerelden Cloudflare API ile DNS yönetir; Cloudflare named tunnel da bununla yapılandırılır. Quick Tunnel için gerekmez |
| OpenSSH istemcisi `ssh` | FRP modu veya Colab SSH sorun giderme | FRP sunucusuna bağlanır; `~/.ssh/config` içindeki anahtar, port ve atlama ayarlarını kullanır |

Gereken GPU runtime'larını kullanabilen bir Google/Colab hesabı ve boş alanı olan bir Google Drive hazırlayın. GPU kotası, compute unit bakiyesi ve makine tipi uygunluğu hesabın güncel durumuna bağlıdır. İlk Google, Drive ve Cloudflare yetkilendirmesi tarayıcıda yapılır.

### 1. Colab CLI kurulumu

Önce [uv resmi talimatlarına](https://docs.astral.sh/uv/getting-started/installation/) göre `uv`'yi kurun ve Python 3.12+ hazırlayın. Bağımsız bir araç ortamında kurun:

```bash
uv tool install google-colab-cli
uv tool update-shell
# Terminali yeniden açın ya da update-shell'in söylediği şekilde PATH'i yükleyin
colab version
colab --help
```

Kaynak: [google-colab-cli](https://github.com/googlecolab/google-colab-cli). `colab sessions` ilk çalıştırıldığında CLI'ın yönlendirmesiyle Google kimlik doğrulamasını tamamlayın; kendi OAuth istemci yapılandırmasını vermesi gereken ortamlarda dosya, upstream talimatlarına göre varsayılan `~/.colab-cli-oauth-config.json` konumuna konmalıdır. Bu proje `colab`'ı arka planda çağırdığı için yetkilendirme önceden aynı kullanıcı ortamında yapılmalıdır.

```bash
colab sessions       # Colab erişimini doğrular; yeni VM oluşturmaz
colab usage          # hesabın compute unit kullanımı/bakiyesi
```

### 2. `colab dfs` eklentisinin kurulumu ve yetkilendirilmesi

Bu proje başlatma sırasında varsayılan olarak `colab dfs mount` çağırır, notebook içindeki etkileşimli `drive.mount()`'a dayanmaz. Eklenti [deluxebear/colab-dfs-ext](https://github.com/deluxebear/colab-dfs-ext) deposundandır ve Colab CLI'dan sonra ayrıca kurulmalıdır:

```bash
curl -fsSL https://raw.githubusercontent.com/deluxebear/colab-dfs-ext/main/install.sh | sh
```

Eklenti deposunu klonlayıp `python3 install.py` de çalıştırabilirsiniz. Kurucu kaynağı `~/.colab-dfs-ext/` altına kaydeder ve `colab` komutunu sağlayan Python ortamına otomatik bağlar; ayrıntılı seçenekler için [eklenti README'si](https://github.com/deluxebear/colab-dfs-ext/blob/main/README.md). Kurduktan sonra denetleyin ve tek seferlik Drive yetkilendirmesini yapın:

```bash
colab dfs --help
colab dfs version
colab dfs doctor
colab dfs consent     # tarayıcıda bir kez yetki verin, sonraki bağlamalar etkileşim istemez
colab dfs show        # yetki özetini gösterir, refresh token'ı yazdırmaz
```

Drive yetkilendirmesi ile Colab oturum kimlik doğrulaması ayrı yapılır. Eklentinin kimlik bilgileri `~/.config/colab-cli/dfs_credentials.json` dosyasındadır. Normal kullanımda servis betikleri uzaktaki `/content/drive`'ı otomatik bağlar, `mount`'u elle çağırmak gerekmez; boşta bir oturumla `colab dfs mount -s <oturum-adı>` ile doğrulayabilirsiniz.

Colab CLI'ı güncelledikten sonra eklentinin hâlâ yerinde olduğunu `colab dfs --help` ile kontrol edin; komut çalışıyorsa `colab dfs install` ile yeniden bağlayabilirsiniz. `No such command: dfs` görünür ve eklenti hâlâ `~/.colab-dfs-ext/` altındaysa kurucuyu yeniden çalıştırarak onarın:

```bash
python3 "$HOME/.colab-dfs-ext/install.py"
colab dfs --help
```

Eklenti dizini yoksa yukarıdaki kurulum komutunu yeniden çalıştırın. `up --no-drive` otomatik bağlamayı atlar, ancak Drive'daki ses örnekleri, görev kalıcılığı ve çıktı kaydetme gibi özellikler yine Drive'a bağlıdır; bu özellikler için bir alternatif değildir.

### 3. Cloudflare `cf` CLI kurulumu

Sabit alan adı kullanıldığında, Cloudflare Tunnel ya da FRP fark etmeksizin `cf`'nin **yerel makineye** kurulup yetkilendirilmesi gerekir. Buradaki `cf`, Cloud Foundry CLI veya uzakta çalışan `cloudflared` değil, Cloudflare CLI'dır.

Kurulu `cf` sürümünün gerektirdiği Node.js/npm'i hazırlayıp kurun:

```bash
node --version
npm --version
npm install --global cf
cf --version
cf auth login
cf auth whoami
```

Kurulum ve kimlik doğrulama için [Cloudflare resmi belgeleri](https://developers.cloudflare.com/cf/get-started/). Hesabın hedef alan adı için Zone/DNS yönetim yetkisi olmalı; named tunnel için tünel yönetim yetkisi de gerekir. `cf zones list` alan adının hesapta olup olmadığını gösterir. Yalnızca dinamik Cloudflare URL'si kullanılacaksa bu bölüm atlanabilir.

Yerelde doğrulanmış kombinasyon `google-colab-cli 0.7.5`, `colab dfs 0.1.0`, `cf 1.0.0-beta.26` ve Node.js `26.13.0`'dır. Bu `cf` sürümünün Node.js gereksinimi `^22.23.0 || ^24.22.0 || ^26.12.0`'dır; güncellerken yalnızca ana sürüme bakmak yerine npm paketinin `engines` bildirimini esas alın.

### 4. FRP modu için ek gereksinimler

Yerelde OpenSSH ve sunucuya etkileşimsiz bağlanabilen bir anahtar yapılandırması gerekir:

```bash
ssh -V
ssh -o BatchMode=yes root@frpip 'id -u'
```

`root@frpip` bir örnektir, gerçek SSH hedefiyle değiştirin; ilk bağlantıda sunucunun host parmak izini doğrulayın. Sunucuda Linux, systemd, Python 3 ve root ya da şifresiz sudo gerekir; mevcut FRP TOML yapılandırmasını okumak için sunucuda Python 3.11+ gerekir. Sunucu internete TCP 7000, 80 ve 443'ü açmalıdır. Yapılandırma komutu sunucudaki frps/Caddy'yi kurar veya yeniden kullanır, yerelde önceden kurmak gerekmez; ayrıntılar [FRP işlemleri](https://github.com/deluxebear/colabllm/blob/master/docs/frp.md) belgesinde.

### Başlatma öncesi kontrol

Proje kök dizininde, servisleri gerçekten çalıştıracağınız terminalde şunları çalıştırın:

```bash
python3 --version
command -v colab
colab version
colab dfs --help
colab dfs show
colabllm tunnel-config
# Sabit alan adı / FRP modu için ayrıca kontrol edin
command -v cf
cf --version
cf auth whoami
# FRP modu için ayrıca kontrol edin
ssh -V
colabllm frp-status
```

Kimlik doğrulama dosyaları yerel kullanıcı dizininde, runtime durumu `~/.colabllm/services/` altındadır. Betikleri çalıştıran makine sürekli çevrimiçi olmalı ve uykuya geçmemelidir: arka plan runner'ı `colab exec` bağlantısını sürdürür, bağlantı koparsa Colab oturumu geri alınabilir. Uzakta gereken model bağımlılıkları, `cloudflared`/`frpc` ve kimlik doğrulayan Caddy notebook tarafından kurulur, yerelde yeniden kurmak gerekmez.

## Cloudflare Tunnel yapılandırması

Yapılandırma tek yerden `colabllm tunnel-config` ile yapılır; mevcut ve yeni tüm runtime'lar ve tüm örnekler, yerel `~/.colabllm/tunnel-config.json` dosyasındaki aynı ana alan adı ayarını okur. FRP etkin değilse ve ayar boşsa `up` (HOST verilmeyen `up --domain` dahil) Cloudflare Quick Tunnel'ın rastgele `https://*.trycloudflare.com` URL'sini kullanır, `cf` girişi gerekmez; URL her başlatmada değişebilir. Koddaki eski `example.com` soneki artık otomatik etkinleşmez.

```bash
# Global ana alan adını göster / ayarla / temizle (proje kökündeki tek giriş)
colabllm tunnel-config
colabllm tunnel-config --base-domain example.com
colabllm tunnel-config --clear

# Ana alan adını yalnızca bu başlatma için ver, yapılandırmaya yazma
colabllm longcat-avatar up --base-domain example.com
# -> https://longcat-avatar.example.com
colabllm qwen-flash up -i 2 --base-domain example.com
# -> https://qwen-2.example.com (servis takma adı ve örnek soneki korunur)

# Ana alan adı ayarlıysa: up otomatik olarak sabit alan adını kullanır; dinamik URL de zorlanabilir
colabllm longcat-avatar up
colabllm longcat-avatar up --quick-tunnel

# Tam alan adı, otomatik üretilen servis alan adının yerine geçer
colabllm longcat-avatar up --domain avatar.example.com
```

`COLABSVC_BASE_DOMAIN=example.com` ortam değişkeni de ayarlanabilir. Seçim sırası: `--quick-tunnel` dinamiği zorlar; tam `--domain HOST` > `--base-domain` > ortam değişkeni > global ayar > dinamik URL. Üç başlatma seçeneği birbirini dışlar. Alan adına yalnızca DNS adı yazılır; `https://`, port veya yol eklenmez.

Cloudflare sabit alan adı `cf auth login` ve hesabın o alan adı için DNS Zone yetkisini gerektirir; başlatmada named tunnel otomatik oluşturulur/yeniden kullanılır, ingress ve DNS CNAME ayarlanır. Ayar değişiklikleri yalnızca sonraki başlatmaları etkiler, çalışan servisleri yeniden başlatmaz; mevcut servisleri uygun bir zamanda kendiniz yeniden başlatın. Dinamik moda geçmek eski tünel ve DNS kayıtlarını silmez; `down --delete-tunnel` ile açıkça temizleyin.

## FRP arka ucu (proje geneli)

Varsayılan Cloudflare Tunnel'dır; FRP global olarak yapılandırıldığında tüm runtime'ların yeni başlatmaları otomatik olarak FRP + sunucu tarafı Caddy HTTPS kullanır.

```bash
# Yapılandırıp sunucu tarafını SSH ile otomatik kur (örnek IP, SSH hedefi ve alan adını değiştirin)
colabllm frp-config --server-ip 203.0.113.10 --ssh root@frpip --base-domain example.com
colabllm frp-status
colabllm indextts up

# Bu başlatma için geçersiz kıl / global FRP'yi kapat
colabllm indextts up --transport cloudflare
colabllm indextts up --quick-tunnel
colabllm frp-config --disable
```

FRP sunucusu varsayılan olarak TCP 7000, Caddy 80/443 kullanır; DNS yerel `cf` ile otomatik olarak DNS only ayarlanır. Yapılandırma, otomatik kurulum, kimlik doğrulama ve temizlik için [FRP işlemleri](https://github.com/deluxebear/colabllm/blob/master/docs/frp.md). Mevcut runtime'lar otomatik geçiş yapmaz.

## Sık kullanılan komutlar

`unsloth-studio` ve `ollama` örneğiyle:

```bash
# 1. Servisi başlat (ana alan adı yoksa rastgele geçici *.trycloudflare.com bağlantısı kullanılır)
colabllm unsloth-studio up

# 2. Tüm servislerin ortak ana alan adını ayarla, sonra sabit alan adıyla başlat
colabllm tunnel-config --base-domain example.com
colabllm ollama up --domain
colabllm qwen-flash up --domain     # https://qwen.example.com'a bağlanır (G4 GPU, 800K bağlam + görsel)

# 3. Erişim URL'sini, giriş şifresini ve API token'ını göster
colabllm ollama info --show-secrets

# 4. Canlı logları izle / durumu ve dış erişimi kontrol et
colabllm ollama logs -f
colabllm ollama status

# 5. Servisi durdur ve Colab VM'ini serbest bırak (--delete-tunnel ile Cloudflare tüneli ve DNS kayıtları da silinir)
colabllm ollama down
colabllm ollama down --delete-tunnel
```

## Çoklu örnek (çoklu oturum) eşzamanlılık desteği (tüm runtime'larda)

Tüm runtime betikleri (`nemotron-transcribe.py`, `comfyui.py`, `qwen-flash.py`, `ollama.py`, `unsloth-studio.py` ve `templates/service.py`), **`-i <ID>` / `--instance <ID>`** (veya `CT_INSTANCE=<ID>` ortam değişkeni) ile birden fazla bağımsız Colab GPU oturumu ve tünelini doğrudan oluşturmayı ve yönetmeyi destekler:

| Örnek argümanı | Colab oturum adı (`-s`) | Durum dizini (`~/.colabllm/services/`) | Tünel adı (`--domain`) | Varsayılan alan adı (`--domain`) | Paylaşılan Drive dizini (`CT_DATA_DIR`) |
| --- | --- | --- | --- | --- | --- |
| Varsayılan (`-i` yok) | `<name>` | `~/.colabllm/services/<name>/` | `colab-<name>` | `<host>.example.com` | `/content/drive/MyDrive/colab-llm/<name>` |
| `-i 1` | `<name>-1` | `~/.colabllm/services/<name>-1/` | `colab-<name>-1` | `<host>-1.example.com` | `/content/drive/MyDrive/colab-llm/<name>` (önbellek ve veri paylaşılır) |
| `-i 2` | `<name>-2` | `~/.colabllm/services/<name>-2/` | `colab-<name>-2` | `<host>-2.example.com` | `/content/drive/MyDrive/colab-llm/<name>` (önbellek ve veri paylaşılır) |

```bash
# Herhangi bir runtime'ın birden fazla eşzamanlı örneğini başlat (her birinin kendi Colab GPU VM'i, Cloudflare tüneli ve alan adı olur)
colabllm comfyui up -i 1 --domain               # -> https://comfyui-1.example.com
colabllm comfyui up -i 2 --domain               # -> https://comfyui-2.example.com
colabllm qwen-flash up -i 2 --domain            # -> https://qwen-2.example.com
colabllm ollama up -i 2 --domain                # -> https://ollama-2.example.com
colabllm unsloth-studio up -i 2 --domain        # -> https://unsloth-studio-2.example.com
colabllm nemotron-transcribe up -i 2 --domain   # -> https://transcribe-2.example.com

# Bir servisin tüm örneklerinin durumunu, URL'sini ve API token'ını listele
colabllm <name> ps

# Belirli bir örneğin ayrıntılarını / loglarını / durumunu göster
colabllm <name> info --show-secrets -i 2
colabllm <name> logs -i 2 -f
colabllm <name> status -i 2

# Belirli bir örneği durdur ya da --all ile servisin tüm örneklerini tek seferde durdur
colabllm <name> down -i 2
colabllm <name> down --all
```

## Yeni bir model servisi ekleme

1. Şablonu kopyalayın:
   ```bash
   cp templates/service.py src/colabllm/services/<name>.py
   cp templates/service.ipynb src/colabllm/services/<name>.ipynb
   chmod +x src/colabllm/services/<name>.py
   ```
2. `src/colabllm/services/<name>.py` içindeki `Service(...)` yapılandırmasını düzenleyin (port, `auth="caddy"` veya `"builtin"`, GPU tipi).
3. `src/colabllm/services/<name>.ipynb` içindeki **Cell 1 (kurulum)** ve **Cell 2 (servisi başlatma, `127.0.0.1`'e bağlı)** hücrelerini düzenleyin.
4. Önce `colabllm <name> up --dry-run` ile çalıştırma planını kontrol edin, sonra `up`'ı gerçekten çalıştırın.
5. Aynı değişiklikte `docs/<name>.md`'yi güncelleyin, `python tools/sync_runtime_docs.py` ile skill referansını üretin ve `skills/colabllm/SKILL.md` içindeki runtime dizinini başlatma, çağırma, girdi/çıktı, kimlik doğrulama ve görev kurtarmayı kapsayacak şekilde güncelleyin. Runtime eklemek, silmek veya komut davranışını değiştirmek kullanıcı skill'inin de güncellenmesini gerektirir.
6. Skill örneklerinin gerçek CLI argümanlarıyla tutarlı olduğunu kontrol edin, dağıtım paketini derleyin ve kurulum, runtime keşfi ve skill dışa aktarımını `tools/check_distribution.py` ile doğrulayın; yeni dosya türü eklediyseniz paketleme kurallarını da güncelleyin. Ekleme, runtime ve kullanıcı skill'i kurulu paketten kullanılabildiğinde tamamlanmış sayılır.

Ayrıntılı kurallar ve sorun giderme için [`.agents/skills/colab-llm-service/SKILL.md`](https://github.com/deluxebear/colabllm/blob/master/.agents/skills/colab-llm-service/SKILL.md).

## Notebook'ları değiştirme

`src/colabllm/services/*.ipynb` ve `templates/service.ipynb`, `templates/build_notebooks.py` tarafından üretilir; elle düzenlemeyin.
Ara katman, önbellek, tünel gibi mantıklar `templates/` altında bağımsız modüller olarak durur (ör. `colab_asgi_common.py`, `colab_cache.py`, `tunnel_client.py`),
böylece yerelde birim testi yapılabilir; değiştirdikten sonra `python3 templates/build_notebooks.py` ile yeniden üretin, testler üretilen sonucun commit'lenen içerikle aynı olduğunu doğrular.

## License

Bu proje [MIT License](https://github.com/deluxebear/colabllm/blob/master/LICENSE) ile açık kaynaktır.
Runtime'ların indirdiği üçüncü taraf modeller, imajlar ve yazılımlar kendi lisans ve kullanım koşullarına tabidir; kullanmadan önce kendiniz kontrol edin.
