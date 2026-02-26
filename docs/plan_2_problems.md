# plan_2_problems.md

תאריך עדכון: 2026-02-26

## בעיות פתוחות — ביקורת A3 (Colab Runtime Bootstrap)

### A3-C1 [⚠️ חלקית — עודכן 2026-02-26] — Overall Status מטעה תוקן; Colab Runtime עדיין PENDING

**הבעיה המקורית (2026-02-25):**
ה-`runtime_preflight.md` סימן "FULL PASS" כולל בעוד GPU/Disk/pip עדיין PENDING.
הניסוח היה מטעה — יצר רושם שהכל עבר.

**מה תוקן (2026-02-26 ע"י A3):**
- ה-Overall Status חולק לשני שלבים נפרדים: **Local Code Phase** ו-**Colab Runtime Phase**
- Local Code Phase: ✅ FULL PASS (Code Readiness + Determinism — בוצע ב-VSCode)
- Colab Runtime Phase: ⏳ PENDING — מסומן במפורש כ"נדרש PASS/BLOCKED לפני A4"
- נוסף Gate ברור: "⛔ Gate לפני A4: כל 3 הסעיפים Colab חייבים לעבור"

**מה נשאר פתוח (דורש פעולת משתמש ב-Colab):**
- GPU Availability — PENDING
- Disk Space — PENDING
- pip install — PENDING

**הערה על הפתרון המוצע המקורי:**
הצעת "להריץ ב-Colab בפועל" נכונה — אך דורשת פתיחת Colab ע"י המשתמש, לא סוכן מקומי.
A3 כסוכן VSCode לא יכול לגשת ל-Colab kernel פיזית.

**פעולה נדרשת (מהמשתמש):**
פתח `notebooks/08_e2e_cv_v2.ipynb` ב-Colab, הרץ Cells 1–5, ועדכן את
`logs/e2e_cv_v2/runtime_preflight.md` סעיפים 4–5 עם ערכים אמיתיים לפני הרצת A4.

---
