# Mobile scaffold (Flutter)

A minimal single-screen Flutter app that calls the same `/normalize` API used by the web demo and
Chrome extension. This is a **starter scaffold**, not a published app: no app icons, no platform
build files (`android/`, `ios/`) are included — run `flutter create .` inside `flutter_app/` once
to generate them, then drop `lib/main.dart` in.

## Run
```bash
cd mobile/flutter_app
flutter create .            # generates android/ ios/ web/ etc. around this lib/
flutter pub get
flutter run                 # pick a connected device / emulator
```

By default the app calls `http://10.0.2.2:8000` (the Android emulator's alias for the host
machine's `localhost`) so it reaches the FastAPI server (`uvicorn api.main:app`) running on your
laptop during development. Change `apiBase` in `lib/main.dart`, or type a new base URL in the
in-app settings field, to point at a real deployment or an iOS simulator (`http://127.0.0.1:8000`
works there directly) or a physical device (use your machine's LAN IP).

## Why this isn't further along
Building and testing a real Flutter app requires the Flutter SDK and a device/emulator, which
this repository does not assume you have. `lib/main.dart` is plain, dependency-light Dart
(just `http` + `shared_preferences`) so it's easy to read, adapt, or extend once you do.
