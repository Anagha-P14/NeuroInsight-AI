#!/bin/bash
# ============================================================
# STEP 2: Deploy the serving API to Cloud Run
# Run these commands in Cloud Shell (console.cloud.google.com -> Activate Cloud Shell)
# Cloud Run free tier: 2 million requests/month, scales to zero when idle.
# ============================================================

set -e

PROJECT_ID="alzheimer-501509"
REGION="us-central1"
BUCKET_NAME="alzheimer-501509-models"
SERVICE_NAME="risk-prediction-api"

# Fix 1: Explicitly set the project and silence the regional access boundary warning if possible
gcloud config set project "$PROJECT_ID"

# Fix 2: Pre-emptively enable essential Google Cloud APIs so the script doesn't prompt or fail
echo "🔄 Enabling necessary services APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

# Fix 3: Migrated legacy 'gsutil' to the modern 'gcloud storage' CLI
echo "📦 Checking Cloud Storage bucket..."
gcloud storage buckets create "gs://${BUCKET_NAME}" --location="$REGION" || true

# 2. Build the container image with Cloud Build (free tier: 120 build-min/day)
echo "🧱 Submitting build to Cloud Build..."
gcloud builds submit --tag "gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# 3. Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "gcr.io/${PROJECT_ID}/${SERVICE_NAME}" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 3 \
  --memory 512Mi \
  --set-env-vars "MODEL_BUCKET=${BUCKET_NAME}"

# 4. Grab the deployed URL (you'll need this for the agent + dashboard)
echo "--------------------------------------------------------"
echo "✅ SUCCESS! Service successfully deployed to Cloud Run."
echo "🔗 Service Endpoint:"
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format "value(status.url)"
echo "--------------------------------------------------------"