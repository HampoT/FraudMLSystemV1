#!/bin/bash
# Backup script for Fraud Detection System
# Backs up database, model artifacts, and configurations
#
# Usage:
#   ./backup.sh                    # Full backup to local directory
#   ./backup.sh --s3 bucket-name   # Upload to S3
#   ./backup.sh --gcs bucket-name  # Upload to GCS
#
# Environment variables:
#   DATABASE_URL - PostgreSQL connection string
#   BACKUP_DIR   - Local backup directory (default: ./backups)

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="fraud_ml_backup_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup database
backup_database() {
    log_info "Backing up database..."
    
    if [ -z "${DATABASE_URL:-}" ]; then
        log_warn "DATABASE_URL not set, skipping database backup"
        return
    fi
    
    # Extract connection details from URL
    DB_FILE="${BACKUP_DIR}/${BACKUP_NAME}/database.sql.gz"
    
    if command -v pg_dump &> /dev/null; then
        pg_dump "${DATABASE_URL}" | gzip > "${DB_FILE}"
        log_info "Database backup saved to ${DB_FILE}"
    else
        log_warn "pg_dump not found, skipping database backup"
    fi
}

# Backup model artifacts
backup_artifacts() {
    log_info "Backing up model artifacts..."
    
    ARTIFACTS_DIR="./artifacts"
    if [ -d "${ARTIFACTS_DIR}" ]; then
        tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/artifacts.tar.gz" -C "${ARTIFACTS_DIR}" .
        log_info "Artifacts backup saved"
    else
        log_warn "Artifacts directory not found at ${ARTIFACTS_DIR}"
    fi
}

# Backup configuration files
backup_configs() {
    log_info "Backing up configuration files..."
    
    CONFIG_BACKUP="${BACKUP_DIR}/${BACKUP_NAME}/configs"
    mkdir -p "${CONFIG_BACKUP}"
    
    # List of config files to backup
    CONFIG_FILES=(
        ".env.example"
        "docker-compose.yml"
        "render.yaml"
        "k8s/deployment.yaml"
        "k8s/dashboard.yaml"
        "k8s/postgres.yaml"
        "k8s/redis.yaml"
    )
    
    for file in "${CONFIG_FILES[@]}"; do
        if [ -f "${file}" ]; then
            cp "${file}" "${CONFIG_BACKUP}/"
        fi
    done
    
    log_info "Configuration files backed up"
}

# Create backup manifest
create_manifest() {
    log_info "Creating backup manifest..."
    
    cat > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.json" << EOF
{
    "backup_name": "${BACKUP_NAME}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "components": {
        "database": $([ -f "${BACKUP_DIR}/${BACKUP_NAME}/database.sql.gz" ] && echo 'true' || echo 'false'),
        "artifacts": $([ -f "${BACKUP_DIR}/${BACKUP_NAME}/artifacts.tar.gz" ] && echo 'true' || echo 'false'),
        "configs": $([ -d "${BACKUP_DIR}/${BACKUP_NAME}/configs" ] && echo 'true' || echo 'false')
    }
}
EOF
}

# Upload to S3
upload_to_s3() {
    local bucket="$1"
    log_info "Uploading backup to S3: ${bucket}..."
    
    # Create final archive
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "${BACKUP_DIR}" "${BACKUP_NAME}"
    
    if command -v aws &> /dev/null; then
        aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" "s3://${bucket}/backups/${BACKUP_NAME}.tar.gz"
        log_info "Backup uploaded to s3://${bucket}/backups/${BACKUP_NAME}.tar.gz"
    else
        log_error "AWS CLI not found"
        exit 1
    fi
}

# Upload to GCS
upload_to_gcs() {
    local bucket="$1"
    log_info "Uploading backup to GCS: ${bucket}..."
    
    # Create final archive
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "${BACKUP_DIR}" "${BACKUP_NAME}"
    
    if command -v gsutil &> /dev/null; then
        gsutil cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" "gs://${bucket}/backups/${BACKUP_NAME}.tar.gz"
        log_info "Backup uploaded to gs://${bucket}/backups/${BACKUP_NAME}.tar.gz"
    else
        log_error "gsutil not found"
        exit 1
    fi
}

# Cleanup old backups (keep last 7)
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    cd "${BACKUP_DIR}"
    ls -dt fraud_ml_backup_* 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null || true
    log_info "Cleanup complete"
}

# Restore from backup
restore_database() {
    local backup_file="$1"
    
    log_info "Restoring database from ${backup_file}..."
    
    if [ -z "${DATABASE_URL:-}" ]; then
        log_error "DATABASE_URL not set"
        exit 1
    fi
    
    if command -v psql &> /dev/null; then
        gunzip -c "${backup_file}" | psql "${DATABASE_URL}"
        log_info "Database restored successfully"
    else
        log_error "psql not found"
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting backup: ${BACKUP_NAME}"
    
    backup_database
    backup_artifacts
    backup_configs
    create_manifest
    
    # Handle cloud upload options
    if [ "${1:-}" == "--s3" ] && [ -n "${2:-}" ]; then
        upload_to_s3 "$2"
    elif [ "${1:-}" == "--gcs" ] && [ -n "${2:-}" ]; then
        upload_to_gcs "$2"
    fi
    
    cleanup_old_backups
    
    log_info "Backup complete: ${BACKUP_DIR}/${BACKUP_NAME}"
    log_info "Manifest: ${BACKUP_DIR}/${BACKUP_NAME}/manifest.json"
}

# Run main if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
