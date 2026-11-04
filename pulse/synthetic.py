"""Synthetic multilingual engagement-survey data with known ("gold") labels.

Every comment is built from hand-written sentences in English, Arabic, Urdu and
Hindi. Each sentence carries its own theme, sentiment, urgency and sensitivity
label, so the correct answer for every comment is known. Site-level patterns are
planted on purpose (camp accommodation in Qatar, safety in Iraq, workload on the
Abu Dhabi shutdown, a harassment cluster in one Kuwait team) so the dashboard has
something real to find.

No real people, companies or survey responses are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LANGS = ("en", "ar", "ur", "hi")


@dataclass(frozen=True)
class Concept:
    id: str
    themes: tuple[str, ...]
    sentiment: str  # positive | neutral | negative
    urgency: str
    sensitive: str  # none or a sensitive category
    text: dict[str, str]


def C(id, themes, sentiment, urgency, en, ar, ur, hi, sensitive="none") -> Concept:  # noqa: N802
    return Concept(id, tuple(themes), sentiment, urgency, sensitive, {"en": en, "ar": ar, "ur": ur, "hi": hi})


CONCEPTS: tuple[Concept, ...] = (
    # --- workload
    C("W1", ["workload"], "negative", "medium",
      "We have been short-staffed for months and everyone is covering two jobs.",
      "نعاني من نقص في عدد الموظفين منذ أشهر، وكل واحد منا يقوم بعمل شخصين.",
      "کئی مہینوں سے اسٹاف کم ہے اور ہر بندہ دو لوگوں کا کام کر رہا ہے۔",
      "कई महीनों से स्टाफ़ की कमी है और हर कोई दो लोगों का काम कर रहा है।"),
    C("W2", ["workload"], "negative", "high",
      "The workload on the shutdown project is not sustainable; good engineers are already resigning because of it.",
      "ضغط العمل في مشروع الإيقاف لا يُحتمل، وبدأ مهندسون جيدون بالاستقالة بسببه.",
      "شٹ ڈاؤن پروجیکٹ پر کام کا بوجھ برداشت سے باہر ہے، اچھے انجینئرز اسی وجہ سے استعفیٰ دے رہے ہیں۔",
      "शटडाउन प्रोजेक्ट पर काम का बोझ बर्दाश्त से बाहर है, अच्छे इंजीनियर इसी वजह से इस्तीफ़ा दे रहे हैं।"),
    C("W3", ["workload", "manager_support"], "positive", "low",
      "Work is busy but well planned, and my supervisor makes sure the load is shared fairly.",
      "العمل كثير لكنه مخطط جيداً، ومشرفي يحرص على توزيع المهام بشكل عادل.",
      "کام زیادہ ہے لیکن پلاننگ اچھی ہے، اور میرا سپروائزر کام برابر تقسیم کرتا ہے۔",
      "काम ज़्यादा है लेकिन प्लानिंग अच्छी है, और मेरा सुपरवाइज़र काम बराबर बाँटता है।"),
    # --- pay & benefits
    C("P1", ["pay_benefits"], "negative", "medium",
      "Salaries have not kept up with the cost of living, especially rent.",
      "الرواتب لم تعد تتناسب مع غلاء المعيشة، وخاصة الإيجارات.",
      "تنخواہیں مہنگائی کے حساب سے نہیں بڑھیں، خاص طور پر کرایہ بہت بڑھ گیا ہے۔",
      "तनख़्वाह महँगाई के हिसाब से नहीं बढ़ी, ख़ासकर किराया बहुत बढ़ गया है।"),
    C("P2", ["pay_benefits"], "negative", "high",
      "Our overtime payments have been delayed for three months and nobody explains why.",
      "مستحقات العمل الإضافي متأخرة منذ ثلاثة أشهر ولا أحد يشرح السبب.",
      "ہمارا اوور ٹائم تین مہینے سے نہیں ملا اور کوئی وجہ بھی نہیں بتاتا۔",
      "हमारा ओवरटाइम तीन महीने से नहीं मिला और कोई वजह भी नहीं बताता।"),
    C("P3", ["pay_benefits"], "positive", "low",
      "The medical insurance is good and the claims process is quick.",
      "التأمين الطبي جيد وإجراءات المطالبات سريعة.",
      "میڈیکل انشورنس اچھی ہے اور کلیم جلدی منظور ہو جاتا ہے۔",
      "मेडिकल इंश्योरेंस अच्छा है और क्लेम जल्दी पास हो जाता है।"),
    C("P4", ["pay_benefits"], "neutral", "low",
      "It would help if the criteria for annual increments were shared openly.",
      "سيكون من المفيد لو تم الإعلان بوضوح عن معايير الزيادة السنوية.",
      "اچھا ہو اگر سالانہ انکریمنٹ کا طریقہ کار سب کو واضح طور پر بتایا جائے۔",
      "अच्छा होगा अगर सालाना इंक्रीमेंट के नियम सबको साफ़-साफ़ बताए जाएँ।"),
    # --- direct manager
    C("M1", ["manager_support"], "positive", "low",
      "My manager listens, supports us and gives clear direction.",
      "مديري يستمع إلينا ويدعمنا ويعطي توجيهات واضحة.",
      "میرا مینیجر ہماری بات سنتا ہے، ساتھ دیتا ہے اور واضح ہدایات دیتا ہے۔",
      "मेरे मैनेजर हमारी बात सुनते हैं, साथ देते हैं और साफ़ निर्देश देते हैं।"),
    C("M2", ["manager_support"], "negative", "medium",
      "My supervisor only talks to us when something goes wrong.",
      "المشرف لا يتحدث معنا إلا عندما يحدث خطأ.",
      "سپروائزر صرف تب بات کرتا ہے جب کوئی غلطی ہو جائے۔",
      "सुपरवाइज़र सिर्फ़ तब बात करते हैं जब कोई गलती हो जाए।"),
    C("M3", ["manager_support", "communication"], "negative", "medium",
      "Priorities in our team change every day and the manager does not explain why.",
      "الأولويات في فريقنا تتغير كل يوم والمدير لا يوضح السبب.",
      "ہماری ٹیم کی ترجیحات روز بدلتی ہیں اور مینیجر وجہ نہیں بتاتا۔",
      "हमारी टीम की प्राथमिकताएँ रोज़ बदलती हैं और मैनेजर वजह नहीं बताते।"),
    # --- leadership
    C("L1", ["leadership"], "negative", "medium",
      "Senior management rarely visits the site and does not understand our problems.",
      "الإدارة العليا نادراً ما تزور الموقع ولا تفهم مشاكلنا.",
      "سینئر مینجمنٹ سائٹ پر بہت کم آتی ہے اور ہمارے مسائل نہیں سمجھتی۔",
      "सीनियर मैनेजमेंट साइट पर बहुत कम आता है और हमारी समस्याएँ नहीं समझता।"),
    C("L2", ["leadership"], "positive", "low",
      "I trust the company's direction and I am proud of the projects we deliver.",
      "أثق في توجه الشركة وأفخر بالمشاريع التي ننجزها.",
      "مجھے کمپنی کی سمت پر بھروسہ ہے اور ہم جو پروجیکٹس مکمل کرتے ہیں ان پر فخر ہے۔",
      "मुझे कंपनी की दिशा पर भरोसा है और हम जो प्रोजेक्ट पूरे करते हैं उन पर गर्व है।"),
    # --- career growth
    C("CG1", ["career_growth"], "negative", "medium",
      "I have been in the same grade for five years with no clear path to promotion.",
      "أنا في نفس الدرجة الوظيفية منذ خمس سنوات دون مسار واضح للترقية.",
      "میں پانچ سال سے ایک ہی گریڈ میں ہوں اور پروموشن کا کوئی واضح راستہ نہیں۔",
      "मैं पाँच साल से एक ही ग्रेड में हूँ और प्रमोशन का कोई साफ़ रास्ता नहीं है।"),
    C("CG2", ["career_growth", "respect_inclusion"], "negative", "medium",
      "Promotions go to people who are close to the boss, not to those who perform.",
      "الترقيات تذهب للمقربين من المدير وليس لمن يعمل بجد.",
      "پروموشن ان کو ملتی ہے جو باس کے قریب ہیں، کام کرنے والوں کو نہیں۔",
      "प्रमोशन उन्हें मिलता है जो बॉस के क़रीब हैं, काम करने वालों को नहीं।"),
    C("CG3", ["career_growth"], "positive", "low",
      "I was given the chance to lead a small work package this year, and it really helped my growth.",
      "أُتيحت لي هذا العام فرصة قيادة حزمة عمل صغيرة، وقد ساعدني ذلك كثيراً على التطور.",
      "اس سال مجھے ایک چھوٹا ورک پیکج لیڈ کرنے کا موقع ملا، اس سے میری بہت گروتھ ہوئی۔",
      "इस साल मुझे एक छोटा वर्क पैकेज लीड करने का मौक़ा मिला, इससे मेरी बहुत ग्रोथ हुई।"),
    # --- training
    C("T1", ["training"], "negative", "medium",
      "We have not had any technical training in the last two years.",
      "لم نحصل على أي تدريب فني خلال السنتين الماضيتين.",
      "پچھلے دو سال میں ہمیں کوئی ٹیکنیکل ٹریننگ نہیں ملی۔",
      "पिछले दो साल में हमें कोई टेक्निकल ट्रेनिंग नहीं मिली।"),
    C("T2", ["training"], "positive", "low",
      "The welding certification course was excellent and I learned a lot.",
      "كانت دورة شهادة اللحام ممتازة وتعلمت منها الكثير.",
      "ویلڈنگ سرٹیفکیشن کورس بہت زبردست تھا، بہت کچھ سیکھا۔",
      "वेल्डिंग सर्टिफ़िकेशन कोर्स बहुत बढ़िया था, बहुत कुछ सीखा।"),
    C("T3", ["training"], "neutral", "low",
      "Please offer English classes for site staff.",
      "نرجو توفير دروس في اللغة الإنجليزية لموظفي الموقع.",
      "سائٹ اسٹاف کے لیے انگلش کلاسز شروع کی جائیں۔",
      "साइट स्टाफ़ के लिए इंग्लिश क्लास शुरू की जाएँ।"),
    # --- communication
    C("CM1", ["communication"], "negative", "medium",
      "We hear about important changes through rumours before management tells us.",
      "نسمع عن التغييرات المهمة من الإشاعات قبل أن تخبرنا الإدارة.",
      "اہم تبدیلیوں کا ہمیں افواہوں سے پتا چلتا ہے، مینجمنٹ بعد میں بتاتی ہے۔",
      "ज़रूरी बदलावों का हमें अफ़वाहों से पता चलता है, मैनेजमेंट बाद में बताता है।"),
    C("CM2", ["communication"], "positive", "low",
      "The monthly town halls keep us informed about the company's plans.",
      "الاجتماعات الشهرية العامة تبقينا على اطلاع بخطط الشركة.",
      "ماہانہ ٹاؤن ہال میٹنگز سے ہمیں کمپنی کے پلانز کا پتا رہتا ہے۔",
      "महीने की टाउन हॉल मीटिंग से हमें कंपनी के प्लान की जानकारी मिलती रहती है।"),
    C("CM3", ["communication"], "neutral", "low",
      "Notices should be shared in Urdu, Hindi and Arabic too, not only in English.",
      "يجب نشر الإعلانات باللغات الأردية والهندية والعربية أيضاً، وليس بالإنجليزية فقط.",
      "نوٹس صرف انگلش میں نہیں بلکہ اردو، ہندی اور عربی میں بھی دیے جائیں۔",
      "नोटिस सिर्फ़ इंग्लिश में नहीं, उर्दू, हिंदी और अरबी में भी दिए जाएँ।"),
    # --- recognition
    C("R1", ["recognition"], "negative", "medium",
      "Good work is never acknowledged; only mistakes get noticed.",
      "العمل الجيد لا يُقدَّر أبداً، فقط الأخطاء هي التي يُلتفت إليها.",
      "اچھے کام کی کبھی تعریف نہیں ہوتی، صرف غلطیاں نظر آتی ہیں۔",
      "अच्छे काम की कभी तारीफ़ नहीं होती, सिर्फ़ गलतियाँ दिखती हैं।"),
    C("R2", ["recognition"], "positive", "low",
      "Our team was recognised at the annual awards and it motivated everyone.",
      "تم تكريم فريقنا في الحفل السنوي وكان ذلك دافعاً للجميع.",
      "سالانہ ایوارڈز میں ہماری ٹیم کو سراہا گیا، سب کا حوصلہ بڑھا۔",
      "सालाना अवॉर्ड्स में हमारी टीम को सम्मान मिला, सबका हौसला बढ़ा।"),
    # --- hours & work-life
    C("WL1", ["work_life"], "negative", "medium",
      "We work twelve-hour shifts six days a week and I hardly see my family.",
      "نعمل ورديات من اثنتي عشرة ساعة ستة أيام في الأسبوع، وبالكاد أرى عائلتي.",
      "ہم ہفتے میں چھ دن بارہ بارہ گھنٹے کی شفٹ کرتے ہیں، گھر والوں سے ملنا مشکل ہو گیا ہے۔",
      "हम हफ़्ते में छह दिन बारह-बारह घंटे की शिफ़्ट करते हैं, घरवालों से मिलना मुश्किल हो गया है।"),
    C("WL2", ["work_life"], "negative", "high",
      "My leave has been postponed twice and I have not been home for two years.",
      "تم تأجيل إجازتي مرتين ولم أزر بلدي منذ سنتين.",
      "میری چھٹی دو بار ملتوی ہو چکی ہے اور میں دو سال سے گھر نہیں گیا۔",
      "मेरी छुट्टी दो बार टल चुकी है और मैं दो साल से घर नहीं गया।"),
    C("WL3", ["work_life"], "positive", "low",
      "The new rotation schedule gives us more time to rest.",
      "جدول المناوبة الجديد يمنحنا وقتاً أطول للراحة.",
      "نئے روٹیشن شیڈول سے آرام کا زیادہ وقت ملتا ہے۔",
      "नए रोटेशन शेड्यूल से आराम का ज़्यादा समय मिलता है।"),
    # --- health & safety (non-sensitive)
    C("HS1", ["health_safety"], "positive", "low",
      "Safety standards on this site are strong and supervisors take them seriously.",
      "معايير السلامة في هذا الموقع قوية والمشرفون يأخذونها بجدية.",
      "اس سائٹ پر سیفٹی کے معیار اچھے ہیں اور سپروائزر انہیں سنجیدگی سے لیتے ہیں۔",
      "इस साइट पर सेफ़्टी के मानक अच्छे हैं और सुपरवाइज़र उन्हें गंभीरता से लेते हैं।"),
    # --- respect (non-sensitive)
    C("RI1", ["respect_inclusion"], "positive", "low",
      "Everyone here is treated with respect, whatever their nationality.",
      "الجميع هنا يُعامَلون باحترام بغض النظر عن جنسياتهم.",
      "یہاں ہر کسی کے ساتھ عزت سے پیش آیا جاتا ہے، چاہے وہ کسی بھی ملک کا ہو۔",
      "यहाँ हर किसी के साथ इज़्ज़त से पेश आया जाता है, चाहे वह किसी भी देश का हो।"),
    # --- facilities
    C("F1", ["facilities"], "negative", "high",
      "The camp rooms are overcrowded, eight people to a room, and the air conditioning has been broken for weeks.",
      "غرف السكن مكتظة، ثمانية أشخاص في الغرفة الواحدة، والمكيف معطل منذ أسابيع.",
      "کیمپ کے کمروں میں بہت رش ہے، ایک کمرے میں آٹھ لوگ ہیں اور اے سی کئی ہفتوں سے خراب ہے۔",
      "कैंप के कमरों में बहुत भीड़ है, एक कमरे में आठ लोग हैं और एसी कई हफ़्तों से ख़राब है।"),
    C("F2", ["facilities", "work_life"], "negative", "medium",
      "The site bus leaves at 4:30 in the morning and we lose two hours a day travelling.",
      "حافلة الموقع تنطلق الساعة الرابعة والنصف فجراً ونخسر ساعتين يومياً في الطريق.",
      "سائٹ کی بس صبح ساڑھے چار بجے نکلتی ہے اور روز دو گھنٹے سفر میں ضائع ہوتے ہیں۔",
      "साइट की बस सुबह साढ़े चार बजे निकलती है और रोज़ दो घंटे सफ़र में बर्बाद होते हैं।"),
    C("F3", ["facilities"], "positive", "low",
      "The new accommodation and mess hall are clean and comfortable.",
      "السكن الجديد وصالة الطعام نظيفان ومريحان.",
      "نئی رہائش اور میس ہال صاف ستھرے اور آرام دہ ہیں۔",
      "नया रहने का इंतज़ाम और मेस हॉल साफ़-सुथरे और आरामदायक हैं।"),
    C("F4", ["facilities"], "negative", "medium",
      "The food in the mess is poor and it is the same every day.",
      "الطعام في المطعم سيئ ونفسه كل يوم.",
      "میس کا کھانا اچھا نہیں اور روز ایک جیسا ہوتا ہے۔",
      "मेस का खाना अच्छा नहीं है और रोज़ एक जैसा होता है।"),
    # --- tools & systems
    C("TE1", ["tools_equipment"], "negative", "medium",
      "We waste hours waiting for tools and spare parts from the store.",
      "نضيع ساعات في انتظار العدد وقطع الغيار من المستودع.",
      "ہم اسٹور سے ٹولز اور اسپیئر پارٹس کے انتظار میں گھنٹوں ضائع کرتے ہیں۔",
      "हम स्टोर से टूल्स और स्पेयर पार्ट्स के इंतज़ार में घंटों बर्बाद करते हैं।"),
    C("TE2", ["tools_equipment"], "negative", "medium",
      "The ERP system is slow and we have to fill in the same forms twice.",
      "نظام ERP بطيء ونضطر لتعبئة النماذج نفسها مرتين.",
      "ای آر پی سسٹم بہت سست ہے اور ایک ہی فارم دو بار بھرنا پڑتا ہے۔",
      "ईआरपी सिस्टम बहुत धीमा है और एक ही फ़ॉर्म दो बार भरना पड़ता है।"),
    C("TE3", ["tools_equipment"], "positive", "low",
      "The new digital permit system has made our work much faster.",
      "نظام التصاريح الرقمي الجديد جعل عملنا أسرع بكثير.",
      "نئے ڈیجیٹل پرمٹ سسٹم سے ہمارا کام بہت تیز ہو گیا ہے۔",
      "नए डिजिटल परमिट सिस्टम से हमारा काम बहुत तेज़ हो गया है।"),
    # --- job security
    C("J1", ["job_security"], "negative", "high",
      "The project ends in March and nobody knows if our contracts will be renewed.",
      "المشروع ينتهي في مارس ولا أحد يعرف إن كانت عقودنا ستُجدَّد.",
      "پروجیکٹ مارچ میں ختم ہو رہا ہے اور کسی کو نہیں پتا کہ ہمارے کنٹریکٹ رینیو ہوں گے یا نہیں۔",
      "प्रोजेक्ट मार्च में ख़त्म हो रहा है और किसी को नहीं पता कि हमारे कॉन्ट्रैक्ट रिन्यू होंगे या नहीं।"),
    C("J2", ["job_security"], "negative", "medium",
      "The move from permanent to project-based contracts has made people anxious.",
      "التحول من العقود الدائمة إلى عقود المشاريع جعل الناس قلقين.",
      "مستقل کنٹریکٹ سے پروجیکٹ بیسڈ کنٹریکٹ پر جانے سے لوگ پریشان ہیں۔",
      "परमानेंट कॉन्ट्रैक्ट से प्रोजेक्ट-बेस्ड कॉन्ट्रैक्ट पर जाने से लोग परेशान हैं।"),
    C("J3", ["job_security"], "positive", "low",
      "I feel secure in my job, and the company looked after us during difficult times.",
      "أشعر بالأمان في وظيفتي، والشركة اعتنت بنا في الأوقات الصعبة.",
      "مجھے اپنی نوکری محفوظ لگتی ہے، اور مشکل وقت میں کمپنی نے ہمارا خیال رکھا۔",
      "मुझे अपनी नौकरी सुरक्षित लगती है, और मुश्किल वक़्त में कंपनी ने हमारा ख़याल रखा।"),
    # --- teamwork
    C("TW1", ["teamwork"], "positive", "low",
      "My colleagues are like family; we help each other every day.",
      "زملائي مثل العائلة، نساعد بعضنا كل يوم.",
      "میرے ساتھی گھر والوں کی طرح ہیں، ہم روز ایک دوسرے کی مدد کرتے ہیں۔",
      "मेरे साथी परिवार जैसे हैं, हम रोज़ एक-दूसरे की मदद करते हैं।"),
    C("TW2", ["teamwork"], "negative", "medium",
      "There is very little cooperation between the engineering and construction teams.",
      "التعاون بين فريقي الهندسة والإنشاءات ضعيف جداً.",
      "انجینئرنگ اور کنسٹرکشن ٹیموں کے درمیان تعاون بہت کم ہے۔",
      "इंजीनियरिंग और कंस्ट्रक्शन टीमों के बीच तालमेल बहुत कम है।"),
    # --- wellbeing (non-sensitive)
    C("WB1", ["wellbeing"], "negative", "medium",
      "Stress levels in our department are very high and there is no support.",
      "مستوى التوتر في قسمنا مرتفع جداً ولا يوجد أي دعم.",
      "ہمارے ڈیپارٹمنٹ میں ذہنی دباؤ بہت زیادہ ہے اور کوئی سپورٹ نہیں۔",
      "हमारे डिपार्टमेंट में तनाव बहुत ज़्यादा है और कोई सपोर्ट नहीं है।"),
    C("WB2", ["wellbeing"], "positive", "low",
      "The new counselling helpline is a good step.",
      "خط الاستشارات النفسية الجديد خطوة جيدة.",
      "نئی کاؤنسلنگ ہیلپ لائن ایک اچھا قدم ہے۔",
      "नई काउंसलिंग हेल्पलाइन एक अच्छा क़दम है।"),
    # --- sensitive: must reach a human reviewer
    C("S1", ["health_safety"], "negative", "critical",
      "The scaffolding at Unit 3 has no guardrails and we were told to keep working anyway.",
      "السقالات في الوحدة 3 بلا حواجز حماية، وقيل لنا أن نستمر في العمل رغم ذلك.",
      "یونٹ 3 پر اسکیفولڈنگ کی ریلنگ نہیں ہے اور پھر بھی ہمیں کام جاری رکھنے کو کہا گیا۔",
      "यूनिट 3 पर स्कैफ़ोल्डिंग में रेलिंग नहीं है और फिर भी हमें काम जारी रखने को कहा गया।",
      sensitive="safety_hazard"),
    C("S2", ["health_safety"], "negative", "critical",
      "People are entering confined spaces without gas testing because the permit takes too long.",
      "يدخل العمال إلى الأماكن المغلقة دون فحص الغازات لأن إصدار التصريح يستغرق وقتاً طويلاً.",
      "پرمٹ میں دیر لگتی ہے اس لیے لوگ گیس ٹیسٹنگ کے بغیر کنفائنڈ اسپیس میں جا رہے ہیں۔",
      "परमिट में देर लगती है इसलिए लोग गैस टेस्टिंग के बिना कन्फ़ाइंड स्पेस में जा रहे हैं।",
      sensitive="safety_hazard"),
    C("S3", ["health_safety", "tools_equipment"], "negative", "critical",
      "Our safety harnesses are old and damaged, and requests for new ones are ignored.",
      "أحزمة الأمان لدينا قديمة وتالفة، وطلبات استبدالها يتم تجاهلها.",
      "ہمارے سیفٹی ہارنس پرانے اور خراب ہیں، نئے مانگو تو کوئی سنتا نہیں۔",
      "हमारे सेफ़्टी हार्नेस पुराने और ख़राब हैं, नए माँगो तो कोई सुनता नहीं।",
      sensitive="safety_hazard"),
    C("S4", ["respect_inclusion", "manager_support"], "negative", "critical",
      "A supervisor in our section shouts at workers, insults them and threatens to send them home.",
      "أحد المشرفين في قسمنا يصرخ على العمال ويهينهم ويهددهم بترحيلهم.",
      "ہمارے سیکشن کا ایک سپروائزر ورکرز پر چیختا ہے، گالیاں دیتا ہے اور واپس گھر بھیجنے کی دھمکی دیتا ہے۔",
      "हमारे सेक्शन का एक सुपरवाइज़र वर्कर्स पर चिल्लाता है, गालियाँ देता है और वापस घर भेजने की धमकी देता है।",
      sensitive="harassment"),
    C("S5", ["respect_inclusion"], "negative", "critical",
      "A senior colleague keeps making unwanted comments about my appearance and I don't feel safe reporting it.",
      "زميل أقدم يواصل الإدلاء بتعليقات غير مرغوب فيها عن مظهري، ولا أشعر بالأمان للإبلاغ عن ذلك.",
      "ایک سینئر ساتھی بار بار میری شکل و صورت پر نامناسب باتیں کرتا ہے اور مجھے شکایت کرنے میں ڈر لگتا ہے۔",
      "एक सीनियर सहकर्मी बार-बार मेरे रंग-रूप पर ग़लत टिप्पणियाँ करता है और मुझे शिकायत करने में डर लगता है।",
      sensitive="harassment"),
    C("S6", ["respect_inclusion", "facilities"], "negative", "high",
      "Workers of certain nationalities get better rooms and easier shifts than the rest of us.",
      "العمال من جنسيات معينة يحصلون على غرف أفضل وورديات أسهل من بقيتنا.",
      "کچھ خاص قومیت کے لوگوں کو اچھے کمرے اور آسان شفٹیں ملتی ہیں، باقی ہم سب کو نہیں۔",
      "कुछ ख़ास देशों के लोगों को अच्छे कमरे और आसान शिफ़्ट मिलती हैं, बाक़ी हम सबको नहीं।",
      sensitive="discrimination"),
    C("S7", ["wellbeing"], "negative", "critical",
      "With the pressure and the debt back home, I feel hopeless and I can't sleep anymore.",
      "مع الضغط والديون في بلدي، أشعر باليأس ولم أعد أستطيع النوم.",
      "کام کے دباؤ اور گھر کے قرضوں کی وجہ سے میں بالکل ناامید ہو گیا ہوں، اب نیند بھی نہیں آتی۔",
      "काम के दबाव और घर के क़र्ज़ की वजह से मैं बिल्कुल नाउम्मीद हो गया हूँ, अब नींद भी नहीं आती।",
      sensitive="wellbeing_risk"),
    C("S8", ["ethics_integrity"], "negative", "high",
      "Some subcontractor timesheets are being inflated and everyone knows about it.",
      "يتم تضخيم ساعات العمل في كشوف بعض المقاولين من الباطن والجميع يعلم بذلك.",
      "کچھ سب کنٹریکٹرز کی ٹائم شیٹس میں گھنٹے بڑھا کر لکھے جا رہے ہیں اور سب کو پتا ہے۔",
      "कुछ सब-कॉन्ट्रैक्टर्स की टाइमशीट में घंटे बढ़ा-चढ़ाकर लिखे जा रहे हैं और सबको पता है।",
      sensitive="misconduct"),
    C("S9", ["respect_inclusion"], "negative", "high",
      "People who raised complaints last year were moved to worse shifts afterwards.",
      "الأشخاص الذين قدموا شكاوى العام الماضي نُقلوا بعدها إلى ورديات أسوأ.",
      "جن لوگوں نے پچھلے سال شکایت کی تھی، بعد میں انہیں بری شفٹوں پر بھیج دیا گیا۔",
      "जिन लोगों ने पिछले साल शिकायत की थी, बाद में उन्हें ख़राब शिफ़्ट में डाल दिया गया।",
      sensitive="retaliation"),
)

# Urdu typed in Latin letters ("Roman Urdu"), as many workers write on phones. Same meaning and labels as the
# Urdu-script sentence with the same id. Keyword search and English-only readers usually miss these entirely.
ROMAN_URDU: dict[str, str] = {
    "W1": "Kai mahino se staff kam hai aur har banda do logon ka kaam kar raha hai.",
    "W2": "Shutdown project par kaam ka bojh bardasht se bahar hai, achay engineers isi wajah se resign kar rahe hain.",
    "W3": "Kaam zyada hai lekin planning achi hai, aur mera supervisor kaam barabar taqseem karta hai.",
    "P1": "Tankhwah mehngai ke hisaab se nahi barhi, khaas kar kiraya bohat barh gaya hai.",
    "P2": "Hamara overtime teen mahine se nahi mila aur koi wajah bhi nahi batata.",
    "P3": "Medical insurance achi hai aur claim jaldi approve ho jata hai.",
    "P4": "Acha ho agar salana increment ka tareeqa sab ko saaf saaf bataya jaye.",
    "M1": "Mera manager hamari baat sunta hai, saath deta hai aur clear direction deta hai.",
    "M2": "Supervisor sirf tab baat karta hai jab koi ghalti ho jaye.",
    "M3": "Hamari team ki priorities roz badalti hain aur manager wajah nahi batata.",
    "L1": "Senior management site par bohat kam aati hai aur hamare masail nahi samajhti.",
    "L2": "Mujhe company ki direction par bharosa hai aur apne projects par fakhr hai.",
    "CG1": "Main paanch saal se ek hi grade mein hoon aur promotion ka koi wazeh rasta nahi.",
    "CG2": "Promotion unko milti hai jo boss ke qareeb hain, kaam karne walon ko nahi.",
    "CG3": "Is saal mujhe ek chota work package lead karne ka mauqa mila, is se meri bohat growth hui.",
    "T1": "Pichle do saal mein humein koi technical training nahi mili.",
    "T2": "Welding certification course zabardast tha, bohat kuch seekha.",
    "T3": "Site staff ke liye English classes shuru ki jayen.",
    "CM1": "Zaroori tabdeeliyon ka humein afwahon se pata chalta hai, management baad mein batati hai.",
    "CM2": "Monthly town hall meetings se company ke plans ka pata rehta hai.",
    "CM3": "Notice sirf English mein nahi, Urdu, Hindi aur Arabic mein bhi diye jayen.",
    "R1": "Achay kaam ki kabhi tareef nahi hoti, sirf ghaltiyan nazar aati hain.",
    "R2": "Annual awards mein hamari team ko saraha gaya, sab ka hosla barha.",
    "WL1": "Hum haftay mein chhe din barah barah ghante ki shift karte hain, ghar walon se milna mushkil ho gaya hai.",
    "WL2": "Meri chutti do dafa postpone ho chuki hai aur main do saal se ghar nahi gaya.",
    "WL3": "Naye rotation schedule se aaram ka zyada waqt milta hai.",
    "HS1": "Is site par safety ke standards achay hain aur supervisors inko seriously lete hain.",
    "RI1": "Yahan har kisi ke saath izzat se pesh aaya jata hai, chahe wo kisi bhi mulk ka ho.",
    "F1": "Camp ke kamron mein bohat rush hai, ek kamre mein aath log hain aur AC kai hafton se kharab hai.",
    "F2": "Site ki bus subah saarhe chaar baje nikalti hai aur roz do ghante safar mein zaya hote hain.",
    "F3": "Nayi rehaish aur mess hall saaf suthre aur aaram deh hain.",
    "F4": "Mess ka khana acha nahi aur roz ek jaisa hota hai.",
    "TE1": "Store se tools aur spare parts ke intezar mein ghanton zaya ho jate hain.",
    "TE2": "ERP system bohat slow hai aur ek hi form do dafa bharna parta hai.",
    "TE3": "Naye digital permit system se hamara kaam bohat tez ho gaya hai.",
    "J1": "Project March mein khatam ho raha hai aur kisi ko nahi pata ke hamare contract renew honge ya nahi.",
    "J2": "Permanent se project based contract par jane se log pareshan hain.",
    "J3": "Mujhe apni naukri mehfooz lagti hai, aur mushkil waqt mein company ne hamara khayal rakha.",
    "TW1": "Mere saathi ghar walon ki tarah hain, hum roz ek doosre ki madad karte hain.",
    "TW2": "Engineering aur construction teams ke darmiyan taawun bohat kam hai.",
    "WB1": "Hamare department mein stress bohat zyada hai aur koi support nahi.",
    "WB2": "Nayi counselling helpline acha qadam hai.",
    "S1": "Unit 3 par scaffolding ki railing nahi hai aur phir bhi humein kaam jari rakhne ko kaha gaya.",
    "S2": "Permit mein der lagti hai is liye log gas testing ke baghair confined space mein ja rahe hain.",
    "S3": "Hamare safety harness purane aur kharab hain, naye mango to koi nahi sunta.",
    "S4": "Hamare section ka ek supervisor workers par cheekhta hai, galiyan deta hai aur wapas ghar bhejne ki dhamki deta hai.",
    "S5": "Ek senior colleague baar baar meri shakal surat par ghalat baatein karta hai aur mujhe shikayat karne mein dar lagta hai.",
    "S6": "Kuch khaas mulkon ke logon ko achay kamre aur aasan shifts milti hain, baaqi hum sab ko nahi.",
    "S7": "Kaam ke pressure aur ghar ke qarzon ki wajah se main bilkul na-umeed ho gaya hoon, ab neend bhi nahi aati.",
    "S8": "Kuch subcontractors ki timesheets mein ghante barha kar likhe ja rahe hain aur sab ko pata hai.",
    "S9": "Jin logon ne pichle saal shikayat ki thi, baad mein unko buri shifts par bhej diya gaya.",
}

NO_CONTENT = {"en": ["No comments.", "N/A", "Nothing to add."], "ar": ["لا يوجد.", "لا تعليق."],
              "ur": ["کوئی تبصرہ نہیں۔", "کچھ نہیں۔"], "hi": ["कोई टिप्पणी नहीं।", "कुछ नहीं।"],
              "ur_rom": ["Koi comment nahi.", "Kuch nahi."]}

CONTACT = {
    "en": "You can reach me on +971 50 123 4567 if you want details.",
    "ar": "يمكنكم التواصل معي على 0501234567 لمزيد من التفاصيل.",
    "ur": "تفصیل کے لیے مجھ سے 0300-1234567 پر رابطہ کریں۔",
    "hi": "ज़्यादा जानकारी के लिए मुझसे 98765 43210 पर संपर्क करें।",
    "ur_rom": "Detail ke liye mujhse 0300-1234567 par rabta karein.",
}
ROMAN_URDU_SHARE = 0.35  # share of Urdu speakers who type in Latin letters
INFORMAL_ENGLISH_SHARE = 0.25

BY_ID = {c.id: c for c in CONCEPTS}
SENSITIVE_IDS = [c.id for c in CONCEPTS if c.sensitive != "none"]
SENSITIVE_WEIGHTS = {"S1": .12, "S2": .10, "S3": .12, "S4": .15, "S5": .10, "S6": .12, "S7": .10, "S8": .09, "S9": .10}
SAFETY_IDS = ["S1", "S2", "S3"]

BASE_THEME_WEIGHT = {
    "workload": 1.0, "pay_benefits": 1.3, "manager_support": 1.0, "leadership": 0.7, "career_growth": 1.0,
    "training": 0.7, "communication": 0.8, "recognition": 0.6, "work_life": 1.0, "health_safety": 0.4,
    "respect_inclusion": 0.25, "facilities": 0.8, "tools_equipment": 0.7, "job_security": 0.6, "teamwork": 0.6,
    "wellbeing": 0.4,
}


@dataclass(frozen=True)
class Site:
    country: str
    site: str
    unit: str
    headcount: int
    langs: dict[str, float]
    boost: dict[str, float]
    positivity: float
    sensitive_rate: float
    safety_heavy: bool = False


SITES: tuple[Site, ...] = (
    Site("Pakistan", "Lahore HQ", "Engineering", 450, {"ur": .55, "en": .45}, {"career_growth": 2.5, "pay_benefits": 1.8}, .35, .012),
    Site("Pakistan", "Karachi Fabrication Yard", "Fabrication", 300, {"ur": .75, "en": .25}, {"tools_equipment": 2.0, "health_safety": 1.3}, .30, .02),
    Site("Saudi Arabia", "Jubail Construction", "Construction", 500, {"ar": .30, "ur": .30, "hi": .20, "en": .20}, {"work_life": 2.5, "pay_benefits": 2.0}, .25, .02),
    Site("Saudi Arabia", "Riyadh Office", "Projects", 150, {"ar": .45, "en": .40, "ur": .15}, {"leadership": 1.6, "communication": 1.6}, .40, .01),
    Site("UAE", "Abu Dhabi Shutdown", "Maintenance", 350, {"en": .35, "hi": .25, "ur": .20, "ar": .20}, {"workload": 3.0, "work_life": 1.5}, .30, .015),
    Site("Qatar", "Ras Laffan Camp", "Construction", 400, {"hi": .40, "ur": .35, "en": .15, "ar": .10}, {"facilities": 4.0}, .25, .02),
    Site("Oman", "Sohar Plant", "Operations", 200, {"ar": .40, "en": .25, "hi": .20, "ur": .15}, {"recognition": 1.6, "training": 1.6}, .60, .01),
    Site("Kuwait", "Ahmadi Refinery Site", "Maintenance", 250, {"ar": .35, "hi": .25, "ur": .20, "en": .20}, {"manager_support": 1.5}, .35, .015),
    Site("Bahrain", "Sitra Office", "Engineering", 100, {"en": .45, "ar": .35, "hi": .10, "ur": .10}, {"job_security": 2.5}, .40, .01),
    Site("Iraq", "Basra Field Site", "Construction", 200, {"ar": .70, "en": .30}, {"health_safety": 2.0}, .30, .07, True),
    Site("Malaysia", "Kuala Lumpur Office", "Engineering", 100, {"en": .85, "ur": .05, "hi": .05, "ar": .05}, {"teamwork": 1.5, "career_growth": 1.3}, .45, .01),
)

FIRST = ["Tariq", "Ayesha", "Khalid", "Rajesh", "Faisal", "Sana", "Omar", "Priya", "Hassan", "Noura", "Imran", "Vikram",
         "Bilal", "Fatima", "Suresh", "Yousef", "Anil", "Mariam", "Usman", "Deepak", "Zainab", "Ahmed", "Nadia", "Karthik",
         "Saad", "Laila", "Arjun", "Hamza", "Rania", "Mohan", "Salma", "Kareem", "Ravi", "Hina", "Tamer", "Asif", "Meera",
         "Waleed", "Farah", "Nikhil", "Adnan", "Bushra", "Chandra", "Dina", "Ehsan", "Gaurav", "Huda", "Irfan", "Jamal",
         "Kavita", "Lubna", "Mustafa", "Neha", "Owais", "Pooja", "Qasim", "Rehana", "Sameer", "Tahir", "Uzma", "Varun",
         "Wasim", "Yasmin", "Zubair", "Amir", "Bina", "Danish", "Esha", "Ghazala", "Harish"]
LAST = ["Mehmood", "Siddiqui", "Al-Harbi", "Nair", "Qureshi", "Iqbal", "Al-Saadi", "Menon", "Raza", "Al-Mansoori", "Sheikh",
        "Rao", "Ahmed", "Zahra", "Kumar", "Haddad", "Sharma", "Khalil", "Ghani", "Joshi", "Hussain", "Saleh", "Farooq",
        "Iyer", "Malik", "Nasser", "Pillai", "Butt", "Aziz", "Das"]


def _site_code(site: Site) -> str:
    return "".join(w[0] for w in site.site.split()[:2]).upper() + site.country[:2].upper()


def _gold(concepts: list[Concept]) -> dict[str, str]:
    themes: list[str] = []
    for c in concepts:
        for t in c.themes:
            if t not in themes:
                themes.append(t)
    pols = {c.sentiment for c in concepts if c.sentiment != "neutral"}
    sentiment = "mixed" if pols == {"positive", "negative"} else (pols.pop() if pols else "neutral")
    order = ["low", "medium", "high", "critical"]
    urgency = max((c.urgency for c in concepts), key=order.index)
    sens = [c.sensitive for c in concepts if c.sensitive != "none"]
    return {"themes": ";".join(themes), "sentiment": sentiment, "urgency": urgency, "sensitive_category": sens[0] if sens else "none"}


def generate(seed: int = 42, comment_rate: float = 0.45, scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (survey, gold). ``survey`` looks like a survey export; ``gold`` holds the true label of every comment."""
    rng = np.random.default_rng(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    used_first: set[str] = set()
    names = [n for n in names if not (n.split()[0] in used_first or used_first.add(n.split()[0]))]  # unique first names
    name_i = 0
    rows, gold_rows = [], []
    rid = 0
    harassment_team: str | None = None

    for site in SITES:
        headcount = max(10, int(site.headcount * scale))
        n_teams = max(2, headcount // 45)
        sizes = np.maximum(4, np.round(rng.dirichlet([2.5] * n_teams) * headcount)).astype(int)
        if site.country in ("Saudi Arabia", "Qatar", "Pakistan") and n_teams > 3:
            sizes[-1] = 6  # a very small team, to show the anonymity threshold at work
        for t_i, size in enumerate(sizes, 1):
            team = f"{_site_code(site)}-T{t_i:02d}"
            manager = names[name_i % len(names)]
            name_i += 1
            if site.country == "Kuwait" and harassment_team is None and size >= 30:
                harassment_team = team
            for _ in range(int(size)):
                rid += 1
                level = rng.choice(["Staff", "Supervisor", "Engineer", "Manager"], p=[.55, .2, .2, .05])
                tenure = rng.choice(["<1 year", "1-3 years", "3-5 years", "5+ years"], p=[.2, .35, .25, .2])
                lang = str(rng.choice(list(site.langs), p=np.array(list(site.langs.values())) / sum(site.langs.values())))
                variant = "ur_rom" if lang == "ur" and rng.random() < ROMAN_URDU_SHARE else lang
                row = {
                    "respondent_id": f"R{rid:05d}", "country": site.country, "site": site.site, "business_unit": site.unit,
                    "team": team, "manager_name": manager, "job_level": level, "tenure_band": tenure,
                    "engagement_score": None, "comment": "",
                }
                score = 3.75
                if rng.random() < comment_rate:
                    text, concepts = _comment(rng, site, variant, manager, team == harassment_team)
                    row["comment"] = text
                    if concepts:
                        g = _gold(concepts)
                        n_pos = sum(c.sentiment == "positive" for c in concepts)
                        n_neg = sum(c.sentiment == "negative" for c in concepts)
                        score += 0.45 * n_pos - 0.55 * n_neg - (0.4 if g["urgency"] in ("high", "critical") else 0)
                        gold_rows.append({"comment_id": row["respondent_id"], "language": lang, "script": _script(variant), **g,
                                          "concepts": ";".join(c.id for c in concepts), "text": text})
                    else:
                        gold_rows.append({"comment_id": row["respondent_id"], "language": lang, "script": _script(variant), "themes": "other",
                                          "sentiment": "neutral", "urgency": "low", "sensitive_category": "none",
                                          "concepts": "NONE", "text": text})
                row["engagement_score"] = int(np.clip(np.round(score + rng.normal(0, 0.7)), 1, 5))
                rows.append(row)
    survey = pd.DataFrame(rows)
    gold = pd.DataFrame(gold_rows)
    return survey, gold


def _script(variant: str) -> str:
    return {"en": "Latn", "ur_rom": "Latn", "ar": "Arab", "ur": "Arab", "hi": "Deva"}.get(variant, "")


_INFORMAL = (("please", "pls"), ("because", "bcoz"), (" you ", " u "), ("are ", "r "), ("months", "mnths"))


def _informal(text: str, rng: np.random.Generator) -> str:
    t = text.lower().replace(";", ",")
    for a, b in _INFORMAL:
        if rng.random() < 0.6:
            t = t.replace(a, b)
    return t.replace(". ", " ").rstrip(".")


def _sentence(c: Concept, variant: str) -> str:
    return ROMAN_URDU[c.id] if variant == "ur_rom" else c.text[variant]


def _comment(rng: np.random.Generator, site: Site, lang: str, manager: str, harassment_team: bool) -> tuple[str, list[Concept]]:
    """``lang`` here is the writing variant: en, ar, ur, hi or ur_rom (Roman Urdu)."""
    if rng.random() < 0.07:
        return str(rng.choice(NO_CONTENT[lang])), []
    n_sent = int(rng.choice([1, 2, 3], p=[.5, .35, .15]))
    themes = list(BASE_THEME_WEIGHT)
    w = np.array([BASE_THEME_WEIGHT[t] * site.boost.get(t, 1.0) for t in themes])
    chosen: list[Concept] = []
    for _ in range(n_sent):
        theme = rng.choice(themes, p=w / w.sum())
        r = rng.random()
        pool = [c for c in CONCEPTS if c.sensitive == "none" and c.themes[0] == theme]
        neutral = [c for c in pool if c.sentiment == "neutral"]
        if neutral and r < 0.12:
            options = neutral
        elif r < 0.12 + site.positivity:
            options = [c for c in pool if c.sentiment == "positive"] or pool
        else:
            options = [c for c in pool if c.sentiment == "negative"] or pool
        if site.site == "Ras Laffan Camp" and theme == "facilities" and rng.random() < 0.6:
            options = [BY_ID["F1"]]
        options = [c for c in options if c not in chosen] or options
        chosen.append(options[int(rng.integers(len(options)))])
    sens_rate = 0.22 if harassment_team else site.sensitive_rate
    if rng.random() < sens_rate:
        if harassment_team:
            sid = "S4"
        elif site.safety_heavy and rng.random() < 0.75:
            sid = str(rng.choice(SAFETY_IDS))
        else:
            ids = list(SENSITIVE_WEIGHTS)
            p = np.array([SENSITIVE_WEIGHTS[i] for i in ids])
            sid = str(rng.choice(ids, p=p / p.sum()))
        pos = int(rng.integers(0, len(chosen) + 1))
        chosen.insert(pos, BY_ID[sid])
        if len(chosen) > 3:
            chosen = [c for c in chosen if c.sensitive != "none"] + [c for c in chosen if c.sensitive == "none"][:2]
    # Keep comments to at most three themes, the most a coder (or the model) assigns.
    while len({t for c in chosen for t in c.themes}) > 3 and len(chosen) > 1:
        drop = next((i for i in range(len(chosen) - 1, -1, -1) if chosen[i].sensitive == "none"), len(chosen) - 1)
        chosen.pop(drop)
    parts = [_sentence(c, lang) for c in chosen]
    if lang in ("en", "ur_rom") and chosen and chosen[0].id in ("M1", "M2") and rng.random() < 0.5:
        first = manager.split()[0]
        for a in ("My manager", "My supervisor", "Mera manager", "Supervisor"):
            parts[0] = parts[0].replace(a, first, 1)
    if rng.random() < 0.015:
        parts.append(CONTACT[lang])
    text = " ".join(parts)
    if lang == "en" and rng.random() < INFORMAL_ENGLISH_SHARE:
        text = _informal(text, rng)
    return text, chosen


def human_coded_sample(gold: pd.DataFrame, per_language: int = 50, extra_sensitive: int = 30, seed: int = 7) -> pd.DataFrame:
    """What an HR analyst would code by hand: a random sample per language, enriched with sensitive comments
    so the safety-recall check has enough cases to mean something."""
    rng = np.random.default_rng(seed)
    parts = []
    for lang, grp in gold.groupby("language"):
        take = grp.sample(min(len(grp), per_language), random_state=int(rng.integers(1_000_000)))
        parts.append(take.assign(sample_stratum="random_by_language"))
    sample = pd.concat(parts)
    rest = gold[(gold["sensitive_category"] != "none") & ~gold["comment_id"].isin(sample["comment_id"])]
    if len(rest):
        extra = rest.sample(min(len(rest), extra_sensitive), random_state=seed).assign(sample_stratum="enriched_sensitive")
        sample = pd.concat([sample, extra])
    sample = sample.sample(frac=1, random_state=seed)
    sample["coder"] = "analyst_1"
    return sample[["comment_id", "language", "script", "text", "themes", "sentiment", "urgency", "sensitive_category", "coder", "sample_stratum"]]
