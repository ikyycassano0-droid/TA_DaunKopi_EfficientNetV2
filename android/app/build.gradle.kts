plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    // Kembalikan ke alamat asli agar MainActivity.kt terbaca dan tidak force close
    namespace = "com.example.kopi_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        // ID Aplikasi keren berbahasa Inggris tetap aktif
        applicationId = "com.coffeedoctor.ai"

        // Wajib minimal API 26 untuk tflite_flutter versi 0.12.1
        minSdk = 26
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // Dukungan penuh untuk Emulator PC (x86_64) maupun HP Asli (arm64-v8a)
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86_64")
        }
    }

    // Mencegah file model .tflite dikompresi agar utuh
    androidResources {
        noCompress += listOf("tflite", "lite")
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
            // Mencegah R8 memotong library C++ TFLite pada mode Release APK
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }
}

flutter {
    source = "../.."
}