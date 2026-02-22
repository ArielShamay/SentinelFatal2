# מדריך מלא: תוסף Google Colab ב־VS Code

> **גרסת התוסף:** 0.3.0 | **מפרסם:** Google | **עודכן לאחרונה:** פברואר 2026

---

## מה זה בכלל?

Google Colab הוא שירות Jupyter Notebook מתארח של גוגל. בעזרת התוסף הזה ניתן לחבר את VS Code ישירות לשרתי Colab – כלומר, לכתוב ולהריץ Notebooks ב־VS Code תוך שימוש בחומרת המחשוב של גוגל (כולל GPU ו־TPU) **ללא צורך בדפדפן**.

---

## שלב 1: וידוא שהתוסף מותקן

1. פתח את VS Code.
2. לחץ על אייקון ה-Extensions בסרגל הצד השמאלי (או `Ctrl+Shift+X`).
3. חפש **"Google Colab"** או **"google.colab"**.
4. וודא שהתוסף **Google Colab** של Google מותקן ומופעל (כפתור "Disable" מופיע – סימן שהוא כבר פעיל).

> **שים לב:** האייקון של Colab **אינו** מופיע בסרגל הצד כברירת מחדל. הגישה לתוסף מתבצעת דרך **פתיחת קובץ Notebook** ולא דרך אייקון ייעודי. ראה פרק "פיצ'רים ניסיוניים" להפעלת Activity Bar.

---

## שלב 2: הפעלה ראשונה – Hello World

### יצירה/פתיחה של Notebook

1. צור קובץ חדש עם הסיומת `.ipynb` אחת משתי הדרכים:
   - `Ctrl+Shift+P` → הקלד `New File` → בחר **Jupyter Notebook**
   - לחלופין, צור קובץ בשם `test.ipynb` ישירות בסייר הקבצים.

2. פתח את הקובץ ב־VS Code.

### חיבור לשרת Colab

3. בפינה השמאלית העליונה של ה-Notebook לחץ על **"Select Kernel"**.
4. בחר **Colab** מהרשימה.

   ![בחירת kernel](kernel-selection)

5. יופיעו שתי אפשרויות:
   - **Auto Connect** – התחברות אוטומטית לשרת Colab ברירת מחדל (מומלץ למתחילים).
   - **New Colab Server** – בחירת סוג מכונה ספציפי (GPU/TPU/CPU).

### התחברות עם חשבון Google

6. VS Code יבקש ממך להתחבר עם חשבון Google. לחץ **Allow**.
7. הדפדפן ייפתח אוטומטית לדף OAuth של גוגל.
8. בחר את חשבון Google שלך ואשר את ההרשאות.
9. תועבר בחזרה ל-VS Code אוטומטית.

### הרצת קוד ראשון

10. כתוב בתא הראשון:
    ```python
    print("Hello from Colab!")
    ```
11. לחץ על כפתור ▶ (Run Cell) או `Shift+Enter`.
12. הפלט יופיע מתחת לתא תוך שניות ספורות.

---

## פקודות זמינות (Command Palette)

פתח את לוח הפקודות עם `Ctrl+Shift+P` והקלד `Colab` לצפייה בכל הפקודות:

| פקודה | תיאור |
|-------|--------|
| `Colab: Remove Server` | הסרת שרת Colab שהוקצה |
| `Colab: Sign Out` | התנתקות מ-Colab |
| `Colab: Mount Google Drive to Server...` | הוספת קוד לחיבור Google Drive לשרת |
| `Colab: Open Terminal` | פתיחת טרמינל מחובר לשרת (ניסיוני) |
| `Colab: Mount Server To Workspace...` | עגינת הסרת לסביבת העבודה (ניסיוני) |

---

## עבודה עם קבצים

### חיבור Google Drive

כדי לגשת לקבצים שב-Google Drive שלך מתוך ה-Notebook:

1. `Ctrl+Shift+P` → `Colab: Mount Google Drive to Server...`
2. התוסף יוסיף תא קוד אוטומטית עם הסניפט הבא:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. הרץ את התא ואשר את הגישה.
4. אחרי החיבור, ניתן לגשת לכל קבצי Drive דרך `/content/drive/MyDrive/`.

---

## ניהול שרתים

### הפעלת שרת חדש
- בצע שוב את תהליך "Select Kernel" ובחר **New Colab Server**.
- ניתן להפעיל מספר שרתים במקביל לעבודה על Notebooks שונים.

### הסרת שרת
- `Ctrl+Shift+P` → `Colab: Remove Server`
- לחלופין, לחץ על כפתור Colab בראש ה-Notebook ובחר **Remove Server**.

---

## פיצ'רים ניסיוניים (Experimental Features)

> ניתן להפעיל את הפיצ'רים הבאים דרך הגדרות VS Code (`Ctrl+,` → חפש "colab"). ייתכן שיידרש reload של VS Code לאחר ההפעלה.

### Activity Bar (אייקון בסרגל הצד)
**זה הפתרון לבעיה שתיארת!** הפיצ'ר הזה מופעל כברירת מחדל בגרסה 0.3.0.

אם לא רואים את האייקון:
1. פתח הגדרות: `Ctrl+,`
2. חפש `colab.experimental.activityBar`
3. וודא שהאפשרות **מסומנת (enabled)**.
4. בצע Reload Window: `Ctrl+Shift+P` → `Developer: Reload Window`.

לאחר ההפעלה, יופיע אייקון Colab בסרגל הצד השמאלי שמאפשר:
- צפייה בשרתים הפעילים
- גישה לתיקיית `/content` של כל שרת ישירות מ-VS Code

### העלאת קבצים לשרת (File Upload)
- לחץ קליק ימני על קובץ/תיקיה בסייר הקבצים של VS Code
- בחר **Upload to Colab**
- אם יש מספר שרתים פעילים, בחר לאיזה שרת להעלות

### עגינת שרת לסביבת העבודה (Server Mounting)
- `Ctrl+Shift+P` → `Colab: Mount Server To Workspace...`
- מאפשר לצפות, ליצור, לערוך ולמחוק קבצים בשרת Colab ישירות מסייר הקבצים של VS Code
- לרענון קבצים שהשתנו מחוץ ל-VS Code: לחץ על אייקון הרענון בסרגל ה-Workspace

### טרמינל של Colab
- `Ctrl+Shift+P` → `Colab: Open Terminal`
- פותח טרמינל bash מחובר לשרת Colab
- מאפשר הרצת פקודות shell ישירות על המכונה של גוגל

---

## מה ניתן לעשות ללא מנוי בתשלום (חינם)

Colab **תמיד חינמי לשימוש בסיסי**. הנה מה שכלול בחינם:

| תכונה | חינמי |
|--------|--------|
| Python Notebooks | כן |
| CPU (זמן ריצה מוגבל) | כן |
| GPU בסיסי (T4) | כן, עם הגבלות |
| RAM עד ~12GB | כן |
| אחסון זמני (`/content`) עד ~100GB | כן |
| חיבור Google Drive (15GB) | כן |
| ספריות Python מותקנות מראש (NumPy, Pandas, TensorFlow, PyTorch ועוד) | כן |
| שיתוף Notebooks | כן |
| הרצת קוד Python, R, Julia | כן |

### מגבלות בגרסה החינמית

- **זמן ריצה מוגבל** – כ-12 שעות ברצף מקסימום, לאחריו השרת מושבת.
- **GPU לא מובטח** – אם יש עומס, ייתכן שלא יוקצה GPU.
- **אין background execution** – אם סוגרים את VS Code, הריצה עוצרת.
- **זיכרון RAM נמוך יותר** לעומת גרסאות בתשלום.

### שימושים מומלצים בחינם

- **לימוד מדעי נתונים ו-ML** – ספריות כמו TensorFlow, PyTorch, Scikit-learn מותקנות מראש.
- **Data Analysis** – עיבוד קבצי CSV/Excel עם Pandas.
- **Computer Vision** – עם OpenCV ו-PIL.
- **NLP** – עם Transformers של HuggingFace.
- **Data Visualization** – Matplotlib, Seaborn, Plotly.
- **אבטיפוס מיידי** – בלי להתקין שום דבר מקומית.

---

## Colab בדפדפן לעומת Colab ב-VS Code

| תכונה | Colab בדפדפן | Colab ב-VS Code |
|--------|--------------|-----------------|
| סביבת עריכה | ממשק Colab | VS Code מלא |
| IntelliSense/Autocomplete | בסיסי | מלא (Pylance וכו') |
| Git integration | לא | כן |
| Extensions | לא | כל תוספי VS Code |
| ניפוי שגיאות (Debug) | מוגבל | מלא |
| ניהול קבצים | דרך Drive | ישירות ב-VS Code |

---

## פתרון בעיות נפוצות

### האייקון לא מופיע בסרגל הצד
1. ודא שהתוסף מותקן ומופעל.
2. פעל לפי הוראות "Activity Bar" בפרק הפיצ'רים הניסיוניים.
3. נסה: `Ctrl+Shift+P` → `Developer: Reload Window`.

### לא ניתן להתחבר לשרת
1. ודא שיש חיבור אינטרנט.
2. נסה `Ctrl+Shift+P` → `Colab: Sign Out` ואז התחבר מחדש.
3. בדוק שחשבון Google תקין ב: [colab.research.google.com](https://colab.research.google.com).

### הכרנל מת (Kernel died)
- זמן הריצה החינמי אזל – המתן מספר שעות או שדרג לגרסה בתשלום.
- בשלב הביניים ניתן לנסות שרת חדש: Select Kernel → New Colab Server.

### קוד רץ לאט
- GPU לא הוקצה. נסה ליצור שרת חדש ולבחר **GPU** בעת ההקמה.

---

## קישורים שימושיים

- [Marketplace](https://marketplace.visualstudio.com/items?itemName=google.colab) – דף התוסף
- [User Guide רשמי](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) (אנגלית)
- [דיווח על באגים](https://github.com/googlecolab/colab-vscode/issues/new?template=bug_report.md)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [תוכניות מחיר](https://colab.research.google.com/signup)
