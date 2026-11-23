import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );
  runApp(const CoffeeApp());
}

// ======================================================================
// STRUKTUR DATA HASIL PREPROCESSING CITRA & VALIDASI OBJEK
// ======================================================================
class PreprocessResult {
  final List<List<List<List<double>>>> inputTensor;
  final bool isLeafDetected;
  final double leafCoverage;
  final String validationNote;

  PreprocessResult({
    required this.inputTensor,
    required this.isLeafDetected,
    required this.leafCoverage,
    required this.validationNote,
  });
}

// ======================================================================
// FUNGSI PREPROCESSING CITRA DI BACKGROUND (ISOLATE)
// Mencegah ANR dan Memfilter Objek Non-Daun (Meja, Lantai, Wajah, dll.)
// ======================================================================
PreprocessResult _preprocessAndValidateImageSync(Uint8List imageBytes) {
  final decodedImage = img.decodeImage(imageBytes);
  if (decodedImage == null) {
    return PreprocessResult(
      inputTensor: [],
      isLeafDetected: false,
      leafCoverage: 0.0,
      validationNote: 'Citra tidak dapat diuraikan',
    );
  }

  // Resize citra ke ukuran 384x384 sesuai arsitektur EfficientNetV2-M
  final resizedImage = img.copyResize(decodedImage, width: 384, height: 384);

  int leafPixelCount = 0;
  const int totalPixels = 384 * 384;

  // Tensor input Float [1, 384, 384, 3] dalam rentang [0.0 - 255.0]
  var tensor = List.generate(
    1,
    (b) => List.generate(
      384,
      (y) => List.generate(
        384,
        (x) {
          final pixel = resizedImage.getPixel(x, y);
          final double r = pixel.r.toDouble();
          final double g = pixel.g.toDouble();
          final double b = pixel.b.toDouble();

          // Deteksi klorofil/jaringan daun (daun hijau atau bercak penyakit)
          final bool isGreenChlorophyll =
              (g > r * 1.05) && (g > b * 1.10) && (g > 35.0);
          final bool isDiseaseTissue =
              (r > b * 1.20) && (g > b) && (r >= 60.0);

          if (isGreenChlorophyll || isDiseaseTissue) {
            leafPixelCount++;
          }

          return [r, g, b];
        },
      ),
    ),
  );

  final double leafCoverage = (leafPixelCount / totalPixels) * 100.0;
  // Minimal 12% area foto memuat karakteristik visual daun/jaringan daun
  final bool isLeaf = leafCoverage >= 12.0;

  return PreprocessResult(
    inputTensor: tensor,
    isLeafDetected: isLeaf,
    leafCoverage: leafCoverage,
    validationNote: isLeaf
        ? 'Objek Daun Terdeteksi'
        : 'Objek bukan daun (tidak ditemukan pigmen daun yang cukup)',
  );
}

class CoffeeApp extends StatelessWidget {
  const CoffeeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Coffee Leaf Diagnosis',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF6F8F5),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B432C),
          primary: const Color(0xFF1B432C),
          secondary: const Color(0xFF386641),
          surface: Colors.white,
        ),
      ),
      home: const CoffeeClassifierScreen(),
    );
  }
}

class CoffeeClassifierScreen extends StatefulWidget {
  const CoffeeClassifierScreen({super.key});

  @override
  State<CoffeeClassifierScreen> createState() => _CoffeeClassifierScreenState();
}

class _CoffeeClassifierScreenState extends State<CoffeeClassifierScreen>
    with SingleTickerProviderStateMixin {
  File? _selectedImage;
  String? _predictionResult;
  double _confidenceScore = 0.0;
  bool _isProcessing = false;

  bool _isPickerActive = false;

  Interpreter? _tfliteInterpreter;
  List<String> _classLabels = [];

  late AnimationController _scannerAnimController;
  late Animation<double> _scannerSmoothAnimation;

  @override
  void initState() {
    super.initState();
    _scannerAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _scannerSmoothAnimation = CurvedAnimation(
      parent: _scannerAnimController,
      curve: Curves.easeInOutQuad,
    );

    _initModelAndLabels();
  }

  @override
  void dispose() {
    _scannerAnimController.dispose();
    _tfliteInterpreter?.close();
    super.dispose();
  }

  // 1. Memuat Model TFLite dengan Multi-Threading (4 Core CPU)
  Future<void> _initModelAndLabels() async {
    try {
      final options = InterpreterOptions()..threads = 4;

      _tfliteInterpreter = await Interpreter.fromAsset(
        'assets/models/model_kopi_efficientnetv2.tflite',
        options: options,
      );

      final labelContent =
          await rootBundle.loadString('assets/models/labels.txt');
      _classLabels = labelContent
          .split('\n')
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty)
          .toList();

      if (mounted) setState(() {});
      debugPrint("Model dan Label TFLite Berhasil Dimuat: $_classLabels");
    } catch (e) {
      debugPrint("Gagal memuat model: $e");
    }
  }

  // 2. Memilih Gambar dari Kamera atau Galeri Lokal HP
  Future<void> _getImage(ImageSource source) async {
    if (_isPickerActive) return;
    _isPickerActive = true;

    try {
      HapticFeedback.lightImpact();
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: source,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 85,
      );

      if (pickedFile != null && mounted) {
        debugPrint("Foto dipilih: ${pickedFile.path}");
        setState(() {
          _selectedImage = File(pickedFile.path);
          _predictionResult = null;
          _isProcessing = true;
        });
        _predictDisease(_selectedImage!);
      }
    } catch (e) {
      debugPrint("Info ImagePicker: $e");
    } finally {
      _isPickerActive = false;
    }
  }

  // 3. Menarik Gambar Sampel Bawaan (Local Asset)
  Future<void> _loadSampleAsset(String assetPath) async {
    try {
      final byteData = await rootBundle.load(assetPath);
      final tempDir = await Directory.systemTemp.createTemp();
      final tempFile = File('${tempDir.path}/sample_leaf.jpg');
      await tempFile.writeAsBytes(byteData.buffer.asUint8List());

      if (!mounted) return;
      setState(() {
        _selectedImage = tempFile;
        _predictionResult = null;
        _isProcessing = true;
      });

      _predictDisease(tempFile);
    } catch (e) {
      if (!mounted) return;
      setState(() => _isProcessing = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Foto sampel belum tersedia di $assetPath')),
      );
    }
  }

  // 4. Preprocessing di Background Worker, Validasi & Prediksi Sesuai Colab
  Future<void> _predictDisease(File imageFile) async {
    if (_tfliteInterpreter == null || _classLabels.isEmpty) {
      if (!mounted) return;
      setState(() => _isProcessing = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Row(
            children: [
              Icon(Icons.error_outline, color: Colors.white),
              SizedBox(width: 10),
              Expanded(
                child: Text('Model AI belum siap! Periksa aset tflite.'),
              ),
            ],
          ),
          backgroundColor: Colors.red.shade700,
          behavior: SnackBarBehavior.floating,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
      );
      return;
    }

    try {
      final imageBytes = await imageFile.readAsBytes();

      // Jalankan decode citra di Background Isolate
      final PreprocessResult prep = await compute(
        _preprocessAndValidateImageSync,
        imageBytes,
      );

      if (prep.inputTensor.isEmpty) {
        if (mounted) setState(() => _isProcessing = false);
        return;
      }

      // Saring langsung jika foto bukan objek daun (meja, lantai, wajah, dll.)
      if (!prep.isLeafDetected) {
        debugPrint("Ditolak Filter: ${prep.validationNote}");

        if (!mounted) return;
        setState(() {
          _confidenceScore = 0.0;
          _predictionResult = "Ini bukan daun kopi";
          _isProcessing = false;
        });
        return;
      }

      // Jalankan model AI EfficientNetV2-M
      var outputTensor = List.filled(1 * _classLabels.length, 0.0)
          .reshape([1, _classLabels.length]);

      _tfliteInterpreter!.run(prep.inputTensor, outputTensor);

      // Konversi aman untuk tipe data TFLite
      List<double> rawOutputs = (outputTensor[0] as List)
          .map((e) => (e as num).toDouble())
          .toList();
      debugPrint("NILAI RAW OUTPUT TFLITE: $rawOutputs");

      // ======================================================================
      // LOGIKA AMBANG BATAS PERSIS SEPERTI GOOGLE COLAB
      // ======================================================================
      double maxScore = rawOutputs.reduce(math.max);
      int topIndex = rawOutputs.indexOf(maxScore);

      // Cek apakah output model berupa skor Logit mentah atau Softmax
      bool isLogitOutput = rawOutputs.any((val) => val < 0.0 || val > 1.0) ||
          (rawOutputs.reduce((a, b) => a + b) - 1.0).abs() > 0.08;

      double finalConfidencePercent = 0.0;
      bool isAcceptedAsCoffee = false;

      if (isLogitOutput) {
        // MODE LOGIT: rumus Softmax manual dari Colab
        List<double> expList =
            rawOutputs.map((z) => math.exp(z - maxScore)).toList();
        double sumExp = expList.reduce((a, b) => a + b);
        List<double> softmax = expList.map((e) => e / sumExp).toList();
        finalConfidencePercent = softmax[topIndex] * 100.0;

        // Ambang batas Logit: batas_minimal = 7.0 dari Colab
        const double batasMinimalLogit = 7.0;
        isAcceptedAsCoffee = maxScore >= batasMinimalLogit;

        debugPrint(
            "[MODE LOGIT] Skor Tertinggi: ${maxScore.toStringAsFixed(2)} | Batas: $batasMinimalLogit | Diterima: $isAcceptedAsCoffee | Keyakinan: ${finalConfidencePercent.toStringAsFixed(1)}%");
      } else {
        // MODE SOFTMAX CADANGAN:
        finalConfidencePercent = maxScore * 100.0;
        const double threshold = 75.0;
        isAcceptedAsCoffee = finalConfidencePercent >= threshold;

        debugPrint(
            "[MODE SOFTMAX] Probabilitas: ${finalConfidencePercent.toStringAsFixed(1)}%");
      }

      if (!mounted) return;
      setState(() {
        if (isAcceptedAsCoffee) {
          _confidenceScore = finalConfidencePercent;
          _predictionResult = _classLabels[topIndex];
        } else {
          _confidenceScore = 0.0;
          _predictionResult = "Ini bukan daun kopi";
        }

        _isProcessing = false;
      });
    } catch (e) {
      if (mounted) setState(() => _isProcessing = false);
      debugPrint("Gagal memproses gambar: $e");
    }
  }

  bool get _isHealthy =>
      _predictionResult?.toLowerCase().contains('healthy') ?? false;

  bool get _isNotCoffee =>
      _predictionResult?.toLowerCase().contains('bukan daun kopi') ?? false;

  // ======================================================================
  // 5. DETAIL GEJALA & PENGENDALIAN SESUAI BUKU PANDUAN TEKNIS
  // ======================================================================
  Map<String, dynamic> _getDiseaseDetail(String label) {
    final lower = label.toLowerCase();

    // 1. PENANGANAN BUKAN DAUN KOPI
    if (lower.contains('bukan daun kopi')) {
      return {
        'title': 'Bukan Daun Kopi / Tidak Teridentifikasi',
        'pathogen': 'Spesies Non-Kopi / Objek Tidak Valid',
        'desc':
            'Citra ditolak karena tidak memenuhi ambang batas kecocokan representasi fitur daun kopi (nilai skor di bawah batas minimal model). Model menolak objek ini agar tidak terjadi salah diagnosa pada tanaman selain kopi.',
        'gejala': [
          'Skor kecocokan fitur visual daun kopi di bawah ambang batas validasi (Logit < 7.0).',
          'Terdeteksi daun tanaman lain (seperti daun mangga, daun jeruk, gulma) atau benda bukan daun.',
          'Kualitas pencahayaan kurang memadai atau sudut pemotretan terlalu jauh dari helai daun utama.'
        ],
        'pengendalian': {
          'Panduan Pengambilan Gambar': [
            'Pastikan objek yang difoto adalah daun tanaman kopi asli (Arabika / Robusta).',
            'Arahkan kamera tegak lurus dan fokus pada helai daun dengan jarak 10 - 20 cm.',
            'Hindari latar belakang yang memuat banyak daun tanaman selain kopi.'
          ]
        }
      };
    }

    // 2. DAUN SEHAT
    if (lower.contains('healthy')) {
      return {
        'title': 'Tanaman Kopi Sehat',
        'pathogen': 'Kondisi Prima (Bebas Patogen)',
        'desc':
            'Helai daun dalam kondisi prima, warna hijau mengkilap merata, pertulangan daun kokoh, serta bebas dari spora jamur atau serangan hama.',
        'gejala': [
          'Permukaan atas dan bawah daun bersih mengilap tanpa bercak klorosis.',
          'Pertumbuhan tunas dan cabang baru seimbang dan segar.',
          'Daya fotosintesis optimal untuk pembentukan buah kopi berkualitas.'
        ],
        'pengendalian': {
          'Kultur Teknis': [
            'Pertahankan pemupukan berimbang NPK & pupuk organik sesuai anjuran.',
            'Lakukan pemangkasan tunas air (wiwil) teratur untuk aerasi kanopi tanaman.',
            'Pertahankan pohon pelindung agar daun tidak mengalami cekaman panas ekstrim.'
          ],
          'Sanitasi': [
            'Jaga kebersihan piringan pohon dari gulma sarang serangga pembawa penyakit.',
            'Pastikan saluran drainase air lancar agar area perakaran tidak becek.'
          ]
        }
      };
    }

    // 3. KARAT DAUN (Hemileia vastatrix) - SESUAI MODUL GAMBAR
    if (lower.contains('rust') || lower.contains('karat')) {
      return {
        'title': 'Karat Daun Kopi',
        'pathogen': 'Jamur Hemileia vastatrix',
        'desc':
            'Penyakit utama yang dapat menurunkan produksi kopi sebesar 20 - 70%. Sangat mudah berkembang pada kelembaban rendah dan suhu tinggi, menyebar melalui percikan air hujan, angin, semut, burung, dan manusia.',
        'gejala': [
          'Daun terinfeksi terdapat bercak kuning muda di permukaan bawah daun, kemudian berubah menjadi kuning tua.',
          'Terbentuk serbuk/tepung berwarna jingga/orange (spora jamur) di permukaan bawah daun yang lama-kelamaan menjadi cokelat.',
          'Gejala lanjut: Daun gugur sebelum waktunya, menyebabkan pohon gundul, cabang/ranting mati (dieback), hingga kematian tanaman.',
          'Terutama sangat rentan menyerang Kopi Arabika karena memiliki karakter daun yang tipis.'
        ],
        'pengendalian': {
          'Kultur Teknis': [
            'Penggunaan klon tahan (seperti Komasti, As2K).',
            'Sambung pucuk, rejuvinasi, atau replanting dengan klon tahan.',
            'Penanaman kopi arabika pada ketinggian ≥ 800 m dpl.',
            'Pemupukan berimbang sesuai anjuran dan pengendalian gulma manual.'
          ],
          'Mekanis': [
            'Pengaturan tanaman penaung melalui pemangkasan (naungan maksimal 70% dan cahaya matahari langsung masuk 30%).'
          ],
          'Pestisida Nabati': [
            'Aplikasi pestisida nabati ekstrak biji mahoni atau asap cair (konsentrasi 5%).',
            'Aplikasi pestisida nabati minyak cengkeh (< 1%) dan MS APH.'
          ],
          'Kimiawi': [
            'Aplikasi fungisida berbahan aktif Azoksistrobin dan Tebukonazol yang telah terdaftar dan mendapat izin resmi Menteri Pertanian.'
          ]
        }
      };
    }

    // 4. BERCAK DAUN CERCOSPORA (Cercospora coffeicola) - SESUAI MODUL GAMBAR
    if (lower.contains('cerscospora') ||
        lower.contains('cercospora') ||
        lower.contains('bercak')) {
      return {
        'title': 'Bercak Daun Cercospora (Brown Eye Spot)',
        'pathogen': 'Jamur Cercospora coffeicola',
        'desc':
            'Disebut juga penyakit Mata Katak / Brown Eye Spot. Menyerang daun dan buah kopi, dengan tingkat kerusakan paling parah terjadi pada fase pembibitan/perbenihan.',
        'gejala': [
          'Terdapat bercak-bercak bulat, cokelat kemerahan atau cokelat tua dengan batas yang jelas dan melingkar konsentris.',
          'Pada bercak yang sudah tua, bagian pusat berwarna putih kelabu menyerupai tepung hitam yang merupakan konidium jamur.',
          'Bercak tampak paling jelas dari sebelah atas daun, umumnya berdiameter < 5 mm dan dikelilingi lingkaran kuning bercincin (tanda halo).',
          'Dalam kondisi cuaca lembap bercak membesar; serangan berat menyebabkan daun dan buah rontok massal.'
        ],
        'pengendalian': {
          'Kultur Teknis': [
            'Mengatur kelembaban dengan cara mengurangi frekuensi penyiraman.',
            'Menjarangkan atap naungan pembibitan sehingga sinar matahari langsung dapat masuk.'
          ],
          'Sanitasi': [
            'Sanitasi dengan menggunting daun yang sakit, kemudian dibakar atau dibenamkan ke dalam tanah.'
          ],
          'Kimiawi': [
            'Aplikasi fungisida berbahan aktif Propineb yang telah terdaftar dan mendapat izin resmi Menteri Pertanian.'
          ]
        }
      };
    }

    // 5. BERCAK DAUN PHOMA / HAWAR DAUN (Phoma costaricensis)
    if (lower.contains('phoma')) {
      return {
        'title': 'Bercak Daun Phoma (Hawar Daun & Ranting)',
        'pathogen': 'Jamur Phoma costaricensis',
        'desc':
            'Penyakit hawar daun dan mati pucuk yang umum menyerang kebun kopi dataran tinggi (> 1.000 m dpl). Berkembang sangat cepat saat suhu dingin, angin kencang berhembus, kabut tebal, dan kelembaban tinggi.',
        'gejala': [
          'Bercak berwarna cokelat tua hingga hitam tidak beraturan pada helaian daun dengan garis konsentris gelap (pola papan target).',
          'Tepi daun nekrosis (kering dan rapuh) seperti terbakar, daun mengerut dan mudah sobek saat diterpa angin.',
          'Infeksi meluas dari tangkai daun hingga merusak tunas apikal, menyebabkan mati pucuk ranting muda (twig dieback).',
          'Gugurnya daun muda dan busuknya kuncup bunga/buah muda pada fase pasca-hujan dingin.'
        ],
        'pengendalian': {
          'Kultur Teknis': [
            'Penanaman pohon pematah angin (windbreak seperti Casuarina atau Flemingia) untuk mengurangi luka daun akibat tiupan angin kencang.',
            'Pemberian pupuk Kalium (K) dan Kalsium (Ca) yang cukup untuk mempertebal dinding sel daun.',
            'Hindari aplikasi pupuk Nitrogen (N) berlebih yang membuat jaringan daun sukulen dan rentan terinfeksi.'
          ],
          'Mekanis & Sanitasi': [
            'Pemangkasan sanitasi pada ranting yang mengalami mati pucuk (potong 5 cm di bawah batas jaringan sehat) lalu dibakar di luar kebun.'
          ],
          'Pestisida Nabati / Biologis': [
            'Aplikasi mikroba antagonis Trichoderma sp. pada area tajuk dan perakaran sebagai pelindung biologis.'
          ],
          'Kimiawi': [
            'Aplikasi fungisida protektif berbahan aktif Tembaga (Copper Hydroxide / Copper Oxychloride) menjelang musim hujan dingin.',
            'Gunakan fungisida sistemik berbahan aktif Mankozeb atau Piraklostrobin yang terdaftar di Kementan bila serangan parah.'
          ]
        }
      };
    }

    // DEFAULT JIKA LABEL LAIN
    return {
      'title': label.replaceAll('_', ' ').toUpperCase(),
      'pathogen': 'Patogen Daun Kopi',
      'desc': 'Terdeteksi anomali pada daun kopi. Periksa gejala fisik lebih lanjut.',
      'gejala': ['Terdapat gejala fisik tidak wajar pada helaian daun.'],
      'pengendalian': {
        'Saran': [
          'Isolasi tanaman yang bergejala agar tidak menular.',
          'Konsultasikan sampel daun dengan dinas pertanian setempat.'
        ]
      }
    };
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: CustomScrollView(
          physics: const BouncingScrollPhysics(),
          slivers: [
            // Header Aplikasi
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
                child: Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF1B432C), Color(0xFF386641)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF1B432C).withValues(alpha: 0.2),
                            blurRadius: 10,
                            offset: const Offset(0, 4),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.eco_rounded,
                        color: Colors.white,
                        size: 26,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                'COFFEEDOCTOR',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: 1.3,
                                  color: Colors.green.shade800,
                                ),
                              ),
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: _tfliteInterpreter != null
                                      ? Colors.green.shade100
                                      : Colors.amber.shade100,
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  _tfliteInterpreter != null
                                      ? 'READY'
                                      : 'LOADING',
                                  style: TextStyle(
                                    fontSize: 9,
                                    fontWeight: FontWeight.w800,
                                    color: _tfliteInterpreter != null
                                        ? Colors.green.shade800
                                        : Colors.amber.shade900,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 2),
                          const Text(
                            'Diagnosa Daun Kopi',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w800,
                              color: Color(0xFF192A1D),
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      tooltip: 'Panduan',
                      icon: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.grey.shade200),
                        ),
                        child: Icon(Icons.info_outline_rounded,
                            color: Colors.grey.shade700, size: 20),
                      ),
                      onPressed: () => _showGuideDialog(context),
                    ),
                  ],
                ),
              ),
            ),

            // Konten Diagnosa & Aksi
            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  _buildImageFrame(),
                  const SizedBox(height: 20),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    child: _buildResultSection(),
                  ),
                  const SizedBox(height: 22),
                  _buildActionButtons(),
                  const SizedBox(height: 12),
                  _buildSampleButton(),
                  const SizedBox(height: 26),
                ]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Bingkai Gambar + Laser Scanner
  Widget _buildImageFrame() {
    return Container(
      height: 320,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(26),
        border: Border.all(
          color: _isProcessing
              ? Colors.greenAccent.shade400.withValues(alpha: 0.6)
              : Colors.grey.shade200,
          width: _isProcessing ? 1.8 : 1.0,
        ),
        boxShadow: [
          BoxShadow(
            color: _isProcessing
                ? Colors.greenAccent.shade400.withValues(alpha: 0.15)
                : Colors.black.withValues(alpha: 0.04),
            blurRadius: _isProcessing ? 24 : 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(26),
        child: Stack(
          alignment: Alignment.center,
          children: [
            Positioned.fill(
              child: _selectedImage != null
                  ? Image.file(
                      _selectedImage!,
                      fit: BoxFit.cover,
                      gaplessPlayback: true,
                    )
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            color: const Color(0xFFE8EFE9),
                            shape: BoxShape.circle,
                            border: Border.all(color: const Color(0xFFCDDDD0)),
                          ),
                          child: Icon(
                            Icons.center_focus_strong_rounded,
                            size: 42,
                            color: Colors.green.shade800,
                          ),
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          'Pindai Daun Kopi',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: Color(0xFF1B2E1E),
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Pilih foto dari galeri HP atau ambil dari kamera',
                          style: TextStyle(
                            fontSize: 13,
                            color: Colors.grey.shade500,
                          ),
                        ),
                      ],
                    ),
            ),
            if (_selectedImage != null && _isProcessing)
              Positioned.fill(
                child: Container(
                  color: Colors.black.withValues(alpha: 0.22),
                ),
              ),
            if (_isProcessing)
              AnimatedBuilder(
                animation: _scannerSmoothAnimation,
                builder: (context, child) {
                  final double topPos =
                      _scannerSmoothAnimation.value * (320.0 - 55.0);

                  return Positioned(
                    top: topPos,
                    left: 0,
                    right: 0,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          height: 50,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [
                                Colors.transparent,
                                Colors.greenAccent.shade400
                                    .withValues(alpha: 0.25),
                              ],
                            ),
                          ),
                        ),
                        Container(
                          height: 3.5,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                Colors.transparent,
                                Colors.greenAccent.shade400,
                                Colors.white,
                                Colors.greenAccent.shade400,
                                Colors.transparent,
                              ],
                              stops: const [0.0, 0.2, 0.5, 0.8, 1.0],
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.greenAccent.shade400
                                    .withValues(alpha: 0.9),
                                blurRadius: 16,
                                spreadRadius: 3,
                              ),
                              BoxShadow(
                                color: Colors.white.withValues(alpha: 0.7),
                                blurRadius: 6,
                                spreadRadius: 1,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            if (_isProcessing) _buildTargetReticle(),
            if (_isProcessing)
              Positioned(
                top: 14,
                left: 14,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.65),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color:
                          Colors.greenAccent.shade400.withValues(alpha: 0.7),
                      width: 1.2,
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 7,
                        height: 7,
                        decoration: BoxDecoration(
                          color: Colors.greenAccent.shade400,
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: Colors.greenAccent.shade400,
                              blurRadius: 6,
                              spreadRadius: 1,
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 7),
                      const Text(
                        'AI ANALYZING',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 1.1,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            if (_selectedImage != null && !_isProcessing)
              Positioned(
                top: 14,
                right: 14,
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    onTap: () {
                      HapticFeedback.selectionClick();
                      setState(() {
                        _selectedImage = null;
                        _predictionResult = null;
                        _confidenceScore = 0.0;
                      });
                    },
                    borderRadius: BorderRadius.circular(30),
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.55),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.close_rounded,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTargetReticle() {
    const double length = 22.0;
    const double thickness = 3.0;
    final Color color = Colors.greenAccent.shade400;

    return Stack(
      children: [
        Positioned(
          top: 12,
          left: 12,
          child: Container(
            width: length,
            height: length,
            decoration: BoxDecoration(
              border: Border(
                top: BorderSide(color: color, width: thickness),
                left: BorderSide(color: color, width: thickness),
              ),
            ),
          ),
        ),
        Positioned(
          top: 12,
          right: 12,
          child: Container(
            width: length,
            height: length,
            decoration: BoxDecoration(
              border: Border(
                top: BorderSide(color: color, width: thickness),
                right: BorderSide(color: color, width: thickness),
              ),
            ),
          ),
        ),
        Positioned(
          bottom: 12,
          left: 12,
          child: Container(
            width: length,
            height: length,
            decoration: BoxDecoration(
              border: Border(
                bottom: BorderSide(color: color, width: thickness),
                left: BorderSide(color: color, width: thickness),
              ),
            ),
          ),
        ),
        Positioned(
          bottom: 12,
          right: 12,
          child: Container(
            width: length,
            height: length,
            decoration: BoxDecoration(
              border: Border(
                bottom: BorderSide(color: color, width: thickness),
                right: BorderSide(color: color, width: thickness),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ======================================================================
  // HASIL DIAGNOSA LENGKAP & RAPI DENGAN KARTU PENGENDALIAN
  // ======================================================================
  Widget _buildResultSection() {
    if (_isProcessing) {
      return Container(
        padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(22),
          border: Border.all(color: Colors.green.shade100),
        ),
        child: Column(
          children: [
            SizedBox(
              width: 38,
              height: 38,
              child: CircularProgressIndicator(
                strokeWidth: 3.2,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.green.shade800),
              ),
            ),
            const SizedBox(height: 14),
            const Text(
              'Menganalisis Citra Daun Kopi...',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: Color(0xFF1E3F20),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'EfficientNetV2-M sedang memvalidasi karakteristik daun kopi',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade500),
            ),
          ],
        ),
      );
    }

    if (_predictionResult == null) {
      return Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.grey.shade200),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.tips_and_updates_rounded,
                    color: Colors.amber.shade700, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Panduan Pengambilan Gambar',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF192A1D),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _buildCheckTip('Pastikan objek yang difoto adalah daun kopi.'),
            _buildCheckTip('Arahkan kamera tegak lurus pada area helai daun.'),
            _buildCheckTip('Hindari foto yang buram, blur, atau terlalu gelap.'),
          ],
        ),
      );
    }

    final detail = _getDiseaseDetail(_predictionResult!);
    final Map<String, dynamic> pengendalian =
        detail['pengendalian'] as Map<String, dynamic>;
    final List<dynamic> gejalaList = detail['gejala'] as List<dynamic>;

    Color statusColor;
    Color statusBg;
    String badgeTitle;
    IconData badgeIcon;

    if (_isHealthy) {
      statusColor = const Color(0xFF2B8A3E);
      statusBg = const Color(0xFFEBFBEE);
      badgeTitle = 'Daun Sehat';
      badgeIcon = Icons.check_circle_rounded;
    } else if (_isNotCoffee) {
      statusColor = const Color(0xFFE67700);
      statusBg = const Color(0xFFFFF4E6);
      badgeTitle = 'Bukan Daun Kopi';
      badgeIcon = Icons.help_outline_rounded;
    } else {
      statusColor = const Color(0xFFD93829);
      statusBg = const Color(0xFFFFEBEA);
      badgeTitle = 'Penyakit Terdeteksi';
      badgeIcon = Icons.warning_amber_rounded;
    }

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: statusColor.withValues(alpha: 0.25)),
        boxShadow: [
          BoxShadow(
            color: statusColor.withValues(alpha: 0.08),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 1. KEPALA DIAGNOSA
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'HASIL DIAGNOSA',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                        color: Colors.grey.shade500,
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: statusBg,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(badgeIcon, size: 14, color: statusColor),
                          const SizedBox(width: 5),
                          Text(
                            badgeTitle,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: statusColor,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  detail['title'] as String,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: statusColor,
                    height: 1.2,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  detail['pathogen'] as String,
                  style: TextStyle(
                    fontSize: 13,
                    fontStyle: FontStyle.italic,
                    fontWeight: FontWeight.w600,
                    color: Colors.grey.shade700,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  detail['desc'] as String,
                  style: TextStyle(
                    fontSize: 13,
                    color: Colors.grey.shade800,
                    height: 1.45,
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      _isNotCoffee
                          ? 'Tingkat Kecocokan Fitur Daun'
                          : 'Tingkat Keyakinan (Confidence)',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    Text(
                      _isNotCoffee ? 'Ditolak' : '${_confidenceScore.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w800,
                        color: statusColor,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: _isNotCoffee
                        ? 0.05
                        : (_confidenceScore / 100).clamp(0.0, 1.0),
                    minHeight: 8,
                    backgroundColor: Colors.grey.shade100,
                    valueColor: AlwaysStoppedAnimation<Color>(statusColor),
                  ),
                ),
              ],
            ),
          ),

          const Divider(height: 1, thickness: 1, color: Color(0xFFF1F1F1)),

          // 2. BAGIAN GEJALA & KARAKTERISTIK
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      _isNotCoffee
                          ? Icons.warning_amber_rounded
                          : Icons.biotech_rounded,
                      size: 20,
                      color: _isNotCoffee
                          ? const Color(0xFFE67700)
                          : Colors.blueGrey.shade800,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      _isNotCoffee
                          ? 'Alasan Penolakan Citra'
                          : 'Gejala & Karakteristik Serangan',
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: Color(0xFF1B2E1E),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                ...gejalaList.map((gejala) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          margin: const EdgeInsets.only(top: 5),
                          width: 6,
                          height: 6,
                          decoration: BoxDecoration(
                            color: statusColor,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            gejala as String,
                            style: TextStyle(
                              fontSize: 12.5,
                              color: Colors.grey.shade800,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                }),
              ],
            ),
          ),

          const Divider(height: 1, thickness: 1, color: Color(0xFFF1F1F1)),

          // 3. BAGIAN PENGENDALIAN TERPADU
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      _isNotCoffee
                          ? Icons.lightbulb_outline_rounded
                          : Icons.shield_outlined,
                      size: 20,
                      color: _isNotCoffee
                          ? const Color(0xFFE67700)
                          : Colors.green.shade800,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      _isNotCoffee
                          ? 'Petunjuk Pengambilan Ulang'
                          : 'Rekomendasi Pengendalian Terpadu',
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: Color(0xFF1B2E1E),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                ...pengendalian.entries.map((entry) {
                  return _buildControlCategoryCard(
                    category: entry.key,
                    items: List<String>.from(entry.value),
                  );
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // Komponen Kartu Kategori Pengendalian
  Widget _buildControlCategoryCard({
    required String category,
    required List<String> items,
  }) {
    Color headerBg;
    Color textColor;
    IconData icon;

    switch (category) {
      case 'Kultur Teknis':
        headerBg = const Color(0xFFF59F00);
        textColor = Colors.white;
        icon = Icons.agriculture_rounded;
        break;
      case 'Mekanis':
      case 'Mekanis & Sanitasi':
      case 'Sanitasi':
        headerBg = const Color(0xFF2F9E44);
        textColor = Colors.white;
        icon = Icons.content_cut_rounded;
        break;
      case 'Pestisida Nabati':
      case 'Pestisida Nabati / Biologis':
        headerBg = const Color(0xFF1971C2);
        textColor = Colors.white;
        icon = Icons.spa_rounded;
        break;
      case 'Kimiawi':
        headerBg = const Color(0xFFF08C00);
        textColor = Colors.white;
        icon = Icons.science_rounded;
        break;
      default:
        headerBg = const Color(0xFF495057);
        textColor = Colors.white;
        icon = Icons.info_outline;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFF8F9FA),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: headerBg,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(13),
                bottomRight: Radius.circular(13),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 14, color: textColor),
                const SizedBox(width: 6),
                Text(
                  category,
                  style: TextStyle(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w800,
                    color: textColor,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: items.map((tip) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '• ',
                        style: TextStyle(
                          fontWeight: FontWeight.w800,
                          fontSize: 13,
                          color: Color(0xFF1B432C),
                        ),
                      ),
                      Expanded(
                        child: Text(
                          tip,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey.shade800,
                            height: 1.35,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCheckTip(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.check_circle_outline,
              size: 16, color: Colors.green.shade600),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey.shade600,
                height: 1.3,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // Tombol Aksi: Galeri & Kamera HP
  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: _buildButton(
            icon: Icons.photo_library_outlined,
            title: 'Galeri HP',
            subtitle: 'Pilih File Lokal',
            gradient: const LinearGradient(
              colors: [Color(0xFF2C5E3B), Color(0xFF1B432C)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            onTap: () => _getImage(ImageSource.gallery),
          ),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: _buildButton(
            icon: Icons.camera_alt_outlined,
            title: 'Kamera HP',
            subtitle: 'Ambil Foto',
            gradient: const LinearGradient(
              colors: [Color(0xFF437651), Color(0xFF2A5939)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            onTap: () => _getImage(ImageSource.camera),
          ),
        ),
      ],
    );
  }

  // Tombol Cepat Sampel Uji Bawaan
  Widget _buildSampleButton() {
    return OutlinedButton.icon(
      onPressed: () => _showSampleDialog(context),
      icon: const Icon(Icons.collections_bookmark_outlined, size: 18),
      label: const Text(
        'Uji Menggunakan Sampel Citra Daun',
        style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
      ),
      style: OutlinedButton.styleFrom(
        foregroundColor: const Color(0xFF1B432C),
        side: BorderSide(color: Colors.green.shade700, width: 1.4),
        padding: const EdgeInsets.symmetric(vertical: 12),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
    );
  }

  void _showSampleDialog(BuildContext context) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Pilih Contoh Daun Uji:',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              ListTile(
                leading: const Icon(Icons.check_circle, color: Colors.green),
                title: const Text('Contoh Daun Sehat (Healthy)'),
                onTap: () {
                  Navigator.pop(context);
                  _loadSampleAsset('assets/samples/healthy.jpg');
                },
              ),
              ListTile(
                leading: const Icon(Icons.warning, color: Colors.orange),
                title: const Text('Contoh Karat Daun (Leaf Rust)'),
                onTap: () {
                  Navigator.pop(context);
                  _loadSampleAsset('assets/samples/rust.jpg');
                },
              ),
              ListTile(
                leading: const Icon(Icons.lens, color: Colors.brown),
                title: const Text('Contoh Bercak Daun (Cercospora)'),
                onTap: () {
                  Navigator.pop(context);
                  _loadSampleAsset('assets/samples/cercospora.jpg');
                },
              ),
              ListTile(
                leading: const Icon(Icons.circle, color: Colors.blueGrey),
                title: const Text('Contoh Bercak Phoma'),
                onTap: () {
                  Navigator.pop(context);
                  _loadSampleAsset('assets/samples/phoma.jpg');
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildButton({
    required IconData icon,
    required String title,
    required String subtitle,
    required Gradient gradient,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Ink(
          decoration: BoxDecoration(
            gradient: gradient,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF1B432C).withValues(alpha: 0.24),
                blurRadius: 14,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: Colors.white, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    Text(
                      subtitle,
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.white.withValues(alpha: 0.85),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showGuideDialog(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 30),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                'Tentang CoffeeDoctor',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF192A1D),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Aplikasi ini menggunakan deep learning EfficientNetV2-M beresolusi 384x384 untuk mengenali 4 kondisi daun kopi: Sehat, Karat Daun, Bercak Cercospora, dan Phoma secara instan dan 100% offline (Edge AI / TFLite). Dilengkapi integrasi threshold logit untuk menolak daun asing/bukan daun kopi.',
                style: TextStyle(
                  fontSize: 13,
                  color: Colors.grey.shade600,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 18),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1B432C),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Mengerti',
                      style: TextStyle(fontWeight: FontWeight.w700)),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}