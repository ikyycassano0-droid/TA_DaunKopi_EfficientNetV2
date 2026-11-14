import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );
  runApp(const CoffeeApp());
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

  Interpreter? _tfliteInterpreter;
  List<String> _classLabels = [];

  late AnimationController _scannerAnimController;

  @override
  void initState() {
    super.initState();
    _scannerAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);

    _initModelAndLabels();
  }

  @override
  void dispose() {
    _scannerAnimController.dispose();
    _tfliteInterpreter?.close();
    super.dispose();
  }

  // 1. Memuat Model TFLite dan Label dari Assets
  Future<void> _initModelAndLabels() async {
    try {
      _tfliteInterpreter = await Interpreter.fromAsset(
        'assets/models/model_kopi_efficientnetv2.tflite',
      );

      final labelContent =
          await rootBundle.loadString('assets/models/labels.txt');
      _classLabels = labelContent
          .split('\n')
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty)
          .toList();

      if (mounted) setState(() {});
      debugPrint("✅ Model dan Label TFLite Berhasil Dimuat: $_classLabels");
    } catch (e) {
      debugPrint("❌ Gagal memuat model: $e");
    }
  }

  // 2. Memilih Gambar dari Kamera atau Galeri Lokal HP
  Future<void> _getImage(ImageSource source) async {
    HapticFeedback.lightImpact();
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(
      source: source,
      imageQuality: 95,
    );

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _predictionResult = null;
        _isProcessing = true;
      });
      _predictDisease(_selectedImage!);
    }
  }

  // 3. Menarik Gambar Sampel Bawaan (Local Asset) Tanpa Perlu Galeri
  Future<void> _loadSampleAsset(String assetPath) async {
    try {
      setState(() {
        _predictionResult = null;
        _isProcessing = true;
      });

      final byteData = await rootBundle.load(assetPath);
      final tempDir = await Directory.systemTemp.createTemp();
      final tempFile = File('${tempDir.path}/sample_leaf.jpg');
      await tempFile.writeAsBytes(byteData.buffer.asUint8List());

      setState(() {
        _selectedImage = tempFile;
      });

      _predictDisease(tempFile);
    } catch (e) {
      setState(() => _isProcessing = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Foto sampel belum tersedia di $assetPath')),
      );
    }
  }

  // 4. Preprocessing dan Prediksi Citra (EfficientNetV2-M 384x384)
  Future<void> _predictDisease(File imageFile) async {
    if (_tfliteInterpreter == null) {
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
      img.Image? decodedImage = img.decodeImage(imageBytes);
      if (decodedImage == null) {
        setState(() => _isProcessing = false);
        return;
      }

      // Resize citra ke ukuran 384x384 sesuai input pelatihan model
      img.Image resizedImage =
          img.copyResize(decodedImage, width: 384, height: 384);

      // Konversi pixel ke Float tensor [1, 384, 384, 3] dalam rentang [0.0 - 255.0]
      var inputTensor = List.generate(
        1,
        (b) => List.generate(
          384,
          (y) => List.generate(
            384,
            (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r.toDouble(),
                pixel.g.toDouble(),
                pixel.b.toDouble(),
              ];
            },
          ),
        ),
      );

      // Output tensor probabilitas [1, 4]
      var outputTensor = List.filled(1 * _classLabels.length, 0.0)
          .reshape([1, _classLabels.length]);

      _tfliteInterpreter!.run(inputTensor, outputTensor);

      List<double> probabilities = List<double>.from(outputTensor[0]);

      int topIndex = 0;
      double topProbability = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > topProbability) {
          topProbability = probabilities[i];
          topIndex = i;
        }
      }

      setState(() {
        _predictionResult = _classLabels[topIndex];
        _confidenceScore = topProbability * 100;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() => _isProcessing = false);
      debugPrint("❌ Gagal memproses gambar: $e");
    }
  }

  bool get _isHealthy =>
      _predictionResult?.toLowerCase().contains('healthy') ?? false;

  // 5. Informasi & Solusi Penyakit (Sudah disesuaikan dengan 4 kelas dataset kopi)
  Map<String, dynamic> _getDiseaseDetail(String label) {
    final lower = label.toLowerCase();
    
    if (lower.contains('healthy')) {
      return {
        'title': 'Tanaman Kopi Sehat',
        'desc': 'Daun dalam kondisi prima tanpa gejala infeksi jamur atau serangan hama.',
        'tips': [
          'Pertahankan pemupukan berimbang NPK & pupuk organik.',
          'Jaga aerasi kebun melalui pemangkasan tunas air secara rutin.',
          'Pantau kelembapan tanah di sekitar perakaran tanaman.'
        ],
      };
    } else if (lower.contains('rust') || lower.contains('karat')) {
      return {
        'title': 'Karat Daun (Leaf Rust)',
        'desc': 'Disebabkan oleh jamur Hemileia vastatrix. Terdapat bercak kuning/oranye seperti serbuk spora di bawah daun.',
        'tips': [
          'Kurangi naungan pohon agar sirkulasi udara & sinar matahari optimal.',
          'Aplikasikan fungisida berbahan aktif tembaga (Copper Hydroxide).',
          'Pangkas dan musnahkan daun yang terserang parah.'
        ],
      };
    } else if (lower.contains('cerscospora') || lower.contains('cercospora') || lower.contains('bercak')) {
      return {
        'title': 'Bercak Daun Cercospora (Mata Katak)',
        'desc': 'Disebabkan oleh jamur Cercospora coffeicola. Terdapat bercak cokelat bulat dengan pusat kelabu menyerupai mata katak.',
        'tips': [
          'Tingkatkan nutrisi tanaman, terutama unsur Nitrogen dan Kalium.',
          'Hindari kelembapan berlebih di area tanaman kopi.',
          'Semprotkan fungisida protektif pada fase awal munculnya gejala.'
        ],
      };
    } else if (lower.contains('phoma')) {
      return {
        'title': 'Bercak Daun Phoma (Hawar Daun)',
        'desc': 'Disebabkan oleh jamur Phoma costaricensis. Menghasilkan bercak cokelat gelap tak beraturan dengan garis melingkar konsentris.',
        'tips': [
          'Hindari luka fisik pada daun akibat angin kencang (gunakan pohon pelindung).',
          'Sanitasi kebun dengan membersihkan ranting dan daun mati di sekitar pohon.',
          'Gunakan fungisida sistemik spektrum luas saat musim penghujan tiba.'
        ],
      };
    } else {
      return {
        'title': label.replaceAll('_', ' ').toUpperCase(),
        'desc': 'Terdeteksi anomali pada daun kopi. Periksa gejala fisik lebih lanjut.',
        'tips': [
          'Isolasi tanaman yang bergejala agar tidak menular.',
          'Konsultasikan sampel daun dengan penyuluh pertanian setempat.',
          'Ambil foto ulang dengan pencahayaan yang lebih terang jika ragu.'
        ],
      };
    }
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
                            color: const Color(0xFF1B432C).withOpacity(0.2),
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
                                'COFFEEDOCTOR AI',
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

            // Konten
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
                  // Tombol Cepat Sampel Uji Bawaan
                  _buildSampleButton(),
                  const SizedBox(height: 24),
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
        border: Border.all(color: Colors.grey.shade200),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 18,
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
            if (_isProcessing)
              AnimatedBuilder(
                animation: _scannerAnimController,
                builder: (context, child) {
                  return Positioned(
                    top: _scannerAnimController.value * 270,
                    left: 0,
                    right: 0,
                    child: Container(
                      height: 3,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Colors.transparent,
                            Colors.greenAccent.shade400,
                            Colors.transparent,
                          ],
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.greenAccent.shade400.withOpacity(0.8),
                            blurRadius: 12,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                  );
                },
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
                        color: Colors.black.withOpacity(0.55),
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

  // Hasil Diagnosa
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
              'EfficientNetV2-M sedang menghitung probabilitas penyakit',
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
            _buildCheckTip('Pastikan daun berada di pencahayaan yang cukup.'),
            _buildCheckTip('Arahkan kamera tepat pada area bercak penyakit.'),
            _buildCheckTip('Hindari foto yang buram, blur, atau terlalu gelap.'),
          ],
        ),
      );
    }

    final detail = _getDiseaseDetail(_predictionResult!);
    final Color statusColor =
        _isHealthy ? const Color(0xFF2B8A3E) : const Color(0xFFD93829);
    final Color statusBg =
        _isHealthy ? const Color(0xFFEBFBEE) : const Color(0xFFFFEBEA);

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: statusColor.withOpacity(0.2)),
        boxShadow: [
          BoxShadow(
            color: statusColor.withOpacity(0.06),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'HASIL DIAGNOSA MODEL',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.1,
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
                          Icon(
                            _isHealthy
                                ? Icons.check_circle_rounded
                                : Icons.warning_amber_rounded,
                            size: 14,
                            color: statusColor,
                          ),
                          const SizedBox(width: 5),
                          Text(
                            _isHealthy ? 'Daun Sehat' : 'Penyakit Terdeteksi',
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
                const SizedBox(height: 8),
                Text(
                  detail['title'] as String,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: statusColor,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  detail['desc'] as String,
                  style: TextStyle(
                    fontSize: 13,
                    color: Colors.grey.shade700,
                    height: 1.4,
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Tingkat Keyakinan (Confidence)',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    Text(
                      '${_confidenceScore.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: statusColor,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: _confidenceScore / 100,
                    minHeight: 8,
                    backgroundColor: Colors.grey.shade100,
                    valueColor: AlwaysStoppedAnimation<Color>(statusColor),
                  ),
                ),
              ],
            ),
          ),
          const Divider(height: 1, thickness: 1, color: Color(0xFFF1F1F1)),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 14, 18, 18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.shield_outlined,
                        size: 18, color: Colors.green.shade800),
                    const SizedBox(width: 8),
                    const Text(
                      'Rekomendasi Penanganan',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF1B2E1E),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                ...(detail['tips'] as List<String>).map((tip) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('• ',
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 13)),
                        Expanded(
                          child: Text(
                            tip,
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey.shade700,
                              height: 1.35,
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

  // Tombol Tambahan: Pilih Sampel Cepat (Untuk Uji Coba & Demo Sidang)
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
                color: const Color(0xFF1B432C).withOpacity(0.24),
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
                  color: Colors.white.withOpacity(0.18),
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
                        color: Colors.white.withOpacity(0.85),
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
                'Tentang CoffeeDoctor AI',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF192A1D),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Aplikasi ini menggunakan deep learning EfficientNetV2-M beresolusi 384x384 untuk mengenali 4 kondisi daun kopi: Sehat, Karat Daun, Bercak Cercospora, dan Phoma secara instan dan 100% offline (Edge AI / TFLite).',
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