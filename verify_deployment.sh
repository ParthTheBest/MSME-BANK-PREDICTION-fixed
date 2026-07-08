#!/bin/bash

# Render Deployment Verification Script
# Run: bash verify_deployment.sh

echo "🔍 MSME Bank Prediction — Render Deployment Verification"
echo "=========================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

checks_passed=0
checks_failed=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅${NC} $1"
        ((checks_passed++))
    else
        echo -e "${RED}❌${NC} $1"
        ((checks_failed++))
    fi
}

# 1. Git Repository
echo "📦 Git Repository:"
git rev-parse --git-dir > /dev/null 2>&1
check "Git repository initialized"

git remote -v | grep -q "github.com/ParthTheBest/MSME-BANK-PREDICTION-fixed"
check "GitHub remote configured"

[ -z "$(git status --porcelain)" ]
check "Working tree clean (all committed)"

echo ""

# 2. Critical Files
echo "📄 Critical Files:"
[ -f "render.yaml" ]
check "render.yaml exists"

[ -f "requirements.txt" ]
check "requirements.txt exists"

[ -f "main.py" ]
check "main.py exists"

[ -f ".gitignore" ]
check ".gitignore exists"

grep -q "\.env" .gitignore
check ".env is in .gitignore"

[ ! -f ".env" ] || grep -q "\.env" .gitignore
check ".env is not committed (ignored)"

echo ""

# 3. Model Files
echo "🤖 Model Files:"
[ -f "models/xgb_model.joblib" ]
check "models/xgb_model.joblib"

[ -f "models/calibrator.joblib" ]
check "models/calibrator.joblib"

[ -f "models/shap_explainer.joblib" ]
check "models/shap_explainer.joblib"

[ -f "models/imputer.joblib" ]
check "models/imputer.joblib"

[ -f "models/feature_list.joblib" ]
check "models/feature_list.joblib"

echo ""

# 4. Static & Evaluation Assets
echo "🖼️ Static Assets:"
[ -d "static" ] && [ -f "static/index.html" ]
check "static/index.html (dashboard)"

[ -d "evaluation_report" ] && [ -f "evaluation_report/roc_curves.png" ]
check "evaluation_report/ (charts)"

echo ""

# 5. Dependencies
echo "📚 Dependencies:"
grep -q "fastapi" requirements.txt
check "fastapi in requirements.txt"

grep -q "uvicorn" requirements.txt
check "uvicorn in requirements.txt"

grep -q "gunicorn" requirements.txt
check "gunicorn in requirements.txt"

grep -q "xgboost" requirements.txt
check "xgboost in requirements.txt"

grep -q "python-dotenv" requirements.txt
check "python-dotenv in requirements.txt"

echo ""

# 6. render.yaml Configuration
echo "⚙️ render.yaml Configuration:"
grep -q "fastapi\|gunicorn" render.yaml
check "FastAPI/Gunicorn config in render.yaml"

grep -q "python" render.yaml
check "Python environment configured"

grep -q "ANTHROPIC_API_KEY\|GEMINI_API_KEY" render.yaml
check "API key environment variables defined"

echo ""

# Summary
echo "=========================================================="
echo -e "Results: ${GREEN}✅ Passed: $checks_passed${NC} | ${RED}❌ Failed: $checks_failed${NC}"

if [ $checks_failed -eq 0 ]; then
    echo -e "${GREEN}🚀 All checks passed! Ready for Render deployment.${NC}"
    exit 0
else
    echo -e "${RED}⚠️ Some checks failed. Please fix before deployment.${NC}"
    exit 1
fi
