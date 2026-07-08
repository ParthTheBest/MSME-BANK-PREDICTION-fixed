#!/bin/bash

# Render Auto-Deploy Script
# Usage: bash deploy-to-render.sh

set -e

echo "🚀 MSME Bank Prediction — Auto Deploy to Render"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Render CLI is installed
if ! command -v render &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Render CLI...${NC}"
    npm install -g @render-oss/render-cli
fi

echo -e "${YELLOW}🔑 Enter Render API Key (from https://dashboard.render.com/api):${NC}"
read -s RENDER_API_KEY

if [ -z "$RENDER_API_KEY" ]; then
    echo -e "${RED}❌ API Key required!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}📝 Enter your Service ID or Service Name:${NC}"
read SERVICE_ID

if [ -z "$SERVICE_ID" ]; then
    echo -e "${RED}❌ Service ID required!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}⏳ Triggering deployment...${NC}"

# Deploy using Render API
RESPONSE=$(curl -s -X POST \
  "https://api.render.com/deploy/srv-${SERVICE_ID}" \
  -H "Authorization: Bearer ${RENDER_API_KEY}" \
  -H "Content-Type: application/json")

if echo "$RESPONSE" | grep -q "error"; then
    echo -e "${RED}❌ Deployment failed:${NC}"
    echo "$RESPONSE"
    exit 1
else
    echo -e "${GREEN}✅ Deployment triggered!${NC}"
    echo ""
    echo "Your app will be live shortly at:"
    echo "https://<service-name>.onrender.com/dashboard"
    echo ""
    echo "Monitor deployment at: https://dashboard.render.com"
fi
