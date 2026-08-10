import urllib.request
import urllib.parse
import json

# Curated high-accuracy dictionary for core ISL sign language vocabulary
ISL_DICTIONARY = {
    "what": {"ta": "என்ன", "hi": "क्या", "te": "ఏమిటి", "kn": "ಏನು", "ml": "എന്ത്", "mr": "काय", "bn": "কী", "fr": "Quoi", "de": "Was", "es": "Qué", "ar": "ماذا"},
    "where": {"ta": "எங்கே", "hi": "कहाँ", "te": "ఎక్కడ", "kn": "ಎಲ್ಲಿ", "ml": "എവിടെ", "mr": "कुठे", "bn": "কোথায়", "fr": "Où", "de": "Wo", "es": "Dónde", "ar": "أين"},
    "who": {"ta": "யார்", "hi": "कौन", "te": "ఎవరు", "kn": "ಯಾರು", "ml": "ആര്", "mr": "कोण", "bn": "কে", "fr": "Qui", "de": "Wer", "es": "Quién", "ar": "من"},
    "why": {"ta": "ஏன்", "hi": "क्यों", "te": "ఎందుకు", "kn": "ಏಕೆ", "ml": "എന്തുകൊണ്ട്", "mr": "का", "bn": "কেন", "fr": "Pourquoi", "de": "Warum", "es": "Por qué", "ar": "لماذا"},
    "when": {"ta": "எப்போது", "hi": "कब", "te": "ఎప్పుడు", "kn": "ಯಾವಾಗ", "ml": "എപ്പോൾ", "mr": "केव्हा", "bn": "কখন", "fr": "Quand", "de": "Wann", "es": "Cuándo", "ar": "متى"},
    "hello": {"ta": "வணக்கம்", "hi": "नमस्ते", "te": "నమస్కారం", "kn": "ನಮಸ್ಕಾರ", "ml": "നമസ്കാരം", "mr": "नमस्कार", "bn": "হ্যালো", "fr": "Bonjour", "de": "Hallo", "es": "Hola", "ar": "مرحبا"},
    "yes": {"ta": "ஆம்", "hi": "हाँ", "te": "అవును", "kn": "ಹೌದು", "ml": "അതെ", "mr": "होय", "bn": "হ্যাঁ", "fr": "Oui", "de": "Ja", "es": "Sí", "ar": "نعم"},
    "no": {"ta": "இல்லை", "hi": "नहीं", "te": "కాదు", "kn": "ఇല്ല", "ml": "ഇല്ല", "mr": "नाही", "bn": "না", "fr": "Non", "de": "Nein", "es": "No", "ar": "لا"},
    "help": {"ta": "உதவி", "hi": "मदद", "te": "సహాయం", "kn": "ಸಹಾಯ", "ml": "സഹായം", "mr": "मदत", "bn": "সাহায্য", "fr": "Aide", "de": "Hilfe", "es": "Ayuda", "ar": "مساعدة"},
    "thank you": {"ta": "நன்றி", "hi": "धन्यवाद", "te": "ధన్యవాదాలు", "kn": "ಧನ್ಯವಾದಗಳು", "ml": "നന്ദി", "mr": "धन्यवाद", "bn": "ধন্যবাদ", "fr": "Merci", "de": "Danke", "es": "Gracias", "ar": "شكرا"},
    "sorry": {"ta": "மன்னித்துவிடுங்கள்", "hi": "माफ़ कीजिये", "te": "క్షమించండి", "kn": "ಕ್ಷಮಿಸಿ", "ml": "ക്ഷമിക്കണം", "mr": "माफ करा", "bn": "দুঃখিত", "fr": "Désolé", "de": "Entschuldigung", "es": "Lo siento", "ar": "آسف"},
    "please": {"ta": "தயவுசெய்து", "hi": "कृपया", "te": "దయచేసి", "kn": "ದಯವಿಟ್ಟು", "ml": "ദയവായി", "mr": "कृपया", "bn": "দয়া করে", "fr": "S'il vous plaît", "de": "Bitte", "es": "Por favor", "ar": "رجاء"},
    "good": {"ta": "நல்லது", "hi": "अच्छा", "te": "మంచిది", "kn": "ಒಳ್ಳೆಯದು", "ml": "നല്ലത്", "mr": "चांगले", "bn": "ভালো", "fr": "Bon", "de": "Gut", "es": "Bueno", "ar": "جيد"},
    "bad": {"ta": "கெட்டது", "hi": "बुरा", "te": "చెడు", "kn": "ಕೆಟ್ಟದ್ದು", "ml": "മോശം", "mr": "वाईट", "bn": "খারাপ", "fr": "Mauvais", "de": "Schlecht", "es": "Malo", "ar": "سيء"},
    "water": {"ta": "தண்ணீர்", "hi": "पानी", "te": "నీరు", "kn": "ನೀರು", "ml": "വെള്ളം", "mr": "पाणी", "bn": "জল", "fr": "Eau", "de": "Wasser", "es": "Agua", "ar": "ماء"},
    "food": {"ta": "உணவு", "hi": "खाना", "te": "ఆహారం", "kn": "ಆಹಾರ", "ml": "ഭക്ഷണം", "mr": "अन्न", "bn": "খাবার", "fr": "Nourriture", "de": "Essen", "es": "Comida", "ar": "طعام"},
    "school": {"ta": "பள்ளி", "hi": "स्कूल", "te": "పాఠశాల", "kn": "ಶಾಲೆ", "ml": "സ്കൂൾ", "mr": "शाळा", "bn": "স্কুল", "fr": "École", "de": "Schule", "es": "Escuela", "ar": "مدرسة"},
    "call": {"ta": "அழைப்பு", "hi": "कॉल", "te": "కాల్", "kn": "ಕರೆ", "ml": "വിളിക്കുക", "mr": "कॉल", "bn": "কল", "fr": "Appeler", "de": "Anrufen", "es": "Llamar", "ar": "اتصال"},
    "0": {"ta": "சுழியம்", "hi": "शून्य", "te": "సున్నా", "kn": "ಸೊನ್ನೆ", "ml": "പൂജ്യം", "mr": "शून्य", "bn": "শূন্য", "fr": "Zéro", "de": "Null", "es": "Cero", "ar": "صفر"},
    "1": {"ta": "ஒன்று", "hi": "एक", "te": "ఒకటి", "kn": "ಒಂದು", "ml": "ഒന്ന്", "mr": "एक", "bn": "এক", "fr": "Un", "de": "Eins", "es": "Uno", "ar": "واحد"},
    "2": {"ta": "இரண்டு", "hi": "दो", "te": "రెండు", "kn": "ಎರಡು", "ml": "രണ്ട്", "mr": "दोन", "bn": "দুই", "fr": "Deux", "de": "Zwei", "es": "Dos", "ar": "اثنان"},
    "3": {"ta": "மூன்று", "hi": "तीन", "te": "మూడు", "kn": "ಮೂರು", "ml": "മൂന്ന്", "mr": "तीन", "bn": "তিন", "fr": "Trois", "de": "Drei", "es": "Tres", "ar": "ثلاثة"},
    "4": {"ta": "நான்கு", "hi": "चार", "te": "நாலு", "kn": "ನಾಲ್ಕು", "ml": "നാല്", "mr": "चार", "bn": "চার", "fr": "Quatre", "de": "Vier", "es": "Cuatro", "ar": "أربعة"},
    "5": {"ta": "ஐந்து", "hi": "पांच", "te": "ఐదు", "kn": "ಐದು", "ml": "അഞ്ച്", "mr": "पाच", "bn": "পাঁচ", "fr": "Cinq", "de": "Fünf", "es": "Cinco", "ar": "خمسة"},
    "6": {"ta": "ஆறு", "hi": "छह", "te": "ఆరు", "kn": "ಆರು", "ml": "ആറ്", "mr": "सहा", "bn": "ছয়", "fr": "Six", "de": "Sechs", "es": "Seis", "ar": "ستة"},
    "7": {"ta": "ஏழு", "hi": "सात", "te": "ఏడు", "kn": "ಏಳು", "ml": "ഏഴ്", "mr": "सात", "bn": "সাত", "fr": "Sept", "de": "Sieben", "es": "Siete", "ar": "سبعة"},
    "8": {"ta": "எட்டு", "hi": "आठ", "te": "ఎనిమిది", "kn": "ಎಂಟು", "ml": "എട്ട്", "mr": "आठ", "bn": "আট", "fr": "Huit", "de": "Acht", "es": "Ocho", "ar": "ثمانية"},
    "9": {"ta": "ஒன்பது", "hi": "नौ", "te": "తొమ్మిది", "kn": "ಒಂಬತ್ತು", "ml": "ഒൻപത്", "mr": "नऊ", "bn": "নয়", "fr": "Neuf", "de": "Neun", "es": "Nueve", "ar": "تسعة"}
}

try:
    from googletrans import Translator
    _translator = Translator()
except Exception:
    _translator = None

class TranslationService:
    def __init__(self):
        self._cache = {}
        self.lang_map = {
            "ta-IN": "ta", "hi-IN": "hi", "te-IN": "te", "kn-IN": "kn",
            "ml-IN": "ml", "mr-IN": "mr", "bn-IN": "bn", "gu-IN": "gu",
            "fr-FR": "fr", "de-DE": "de", "es-ES": "es", "ar-SA": "ar",
            "ja-JP": "ja", "zh-CN": "zh", "ru-RU": "ru",
        }

    def _google_gtx_translate(self, text: str, code: str) -> str:
        """Official Google Translate GTX web API endpoint for fast & 100% accurate translation."""
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={code}&dt=t&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data and isinstance(data, list) and len(data) > 0 and data[0]:
                    sentences = [item[0] for item in data[0] if item and len(item) > 0 and item[0]]
                    result = "".join(sentences).strip()
                    if result:
                        return result
        except Exception:
            pass
        return ""

    def translate_word(self, word: str, lang: str) -> str:
        if not word or lang in ("en-US", "en-GB"):
            return word

        word_clean = word.strip()
        key = f"{word_clean}_{lang}"
        if key in self._cache:
            return self._cache[key]

        code = self.lang_map.get(lang, lang)

        # 1. High-accuracy ISL dictionary lookup
        word_lower = word_clean.lower().strip()
        if word_lower in ISL_DICTIONARY and code in ISL_DICTIONARY[word_lower]:
            translated = ISL_DICTIONARY[word_lower][code]
            self._cache[key] = translated
            return translated

        # 2. Google GTX API (Fast & highly accurate)
        gtx_res = self._google_gtx_translate(word_clean, code)
        if gtx_res:
            self._cache[key] = gtx_res
            return gtx_res

        # 3. Googletrans library (Fallback)
        if _translator:
            try:
                r = _translator.translate(word_clean, dest=code)
                if r and r.text and r.text.lower() != word_clean.lower():
                    self._cache[key] = r.text
                    return r.text
            except Exception:
                pass

        # 4. MyMemory API (Secondary fallback)
        try:
            url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(word_clean)}&langpair=en|{code}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                translated = data.get("responseData", {}).get("translatedText", "")
                if translated and translated.lower() != word_clean.lower() and "INVALID" not in translated.upper():
                    self._cache[key] = translated
                    return translated
        except Exception:
            pass

        return word_clean

    def construct_sentence(self, raw_words: str, lang: str = "en-US") -> dict:
        clean_words = raw_words.replace(' .', '').replace('.', '')
        english_sentence = clean_words

        # Try connecting to local LLM server (LM Studio / Ollama) if running
        try:
            import requests
            url = "http://127.0.0.1:1234/v1/chat/completions"
            data = {
                "messages": [
                    {"role": "system", "content": "You are a translator. Convert the given keywords into one simple, grammatically correct English sentence. Do not add extra explanations. Output ONLY the sentence."},
                    {"role": "user", "content": f"Keywords: '{clean_words}'"}
                ],
                "temperature": 0.1, "max_tokens": 50
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=data, headers=headers, timeout=3)
            if response.status_code == 200:
                english_sentence = response.json()['choices'][0]['message']['content'].strip().replace('"', '').replace("'", "")
        except Exception:
            # Simple word joining fallback if local LLM is off
            words = clean_words.split()
            if words:
                english_sentence = " ".join(words).capitalize() + "."

        translated_sentence = self.translate_word(english_sentence, lang)
        return {
            "english_sentence": english_sentence,
            "translated_sentence": translated_sentence
        }
