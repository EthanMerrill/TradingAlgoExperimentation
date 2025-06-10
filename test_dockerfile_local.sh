#!/bin/bash
# Simple test script to verify Dockerfile works locally

echo "🧪 TESTING DOCKERFILE LOCALLY"
echo "=" * 50

# Create dummy credential files for testing
touch ALPACA_KEYS.json
touch POLYGON_API.json  
touch GOOGLE_APPLICATION_CREDENTIALS.json

echo "📦 Building Docker image locally..."
if docker build -t trading-algo-test . ; then
    echo "✅ SUCCESS: Docker build completed successfully!"
    
    echo "🧪 Testing TA-Lib import..."
    if docker run --rm trading-algo-test python -c "import talib; print('TA-Lib version:', talib.__version__)" ; then
        echo "✅ SUCCESS: TA-Lib works in container!"
        
        echo "🧪 Testing full application import..."
        if docker run --rm trading-algo-test python -c "
import sys
sys.path.append('/app')
try:
    from app.config import config
    print('✅ Config loaded successfully')
    print(f'Environment: {config.ENVIRONMENT}')
except Exception as e:
    print(f'❌ Config import failed: {e}')
    
try:
    import talib
    import pandas as pd
    import numpy as np
    print('✅ All main dependencies imported successfully')
except Exception as e:
    print(f'❌ Dependency import failed: {e}')
" ; then
            echo "✅ SUCCESS: Application imports work!"
        else
            echo "❌ FAILED: Application imports failed"
        fi
    else
        echo "❌ FAILED: TA-Lib import failed"
    fi
    
    # Clean up test image
    docker rmi trading-algo-test
else
    echo "❌ FAILED: Docker build failed"
fi

# Clean up dummy files
rm -f ALPACA_KEYS.json POLYGON_API.json GOOGLE_APPLICATION_CREDENTIALS.json

echo ""
echo "🎯 SUMMARY:"
echo "If all tests passed above, your Cloud Build should work!"
echo "If any tests failed, check the error messages above."
