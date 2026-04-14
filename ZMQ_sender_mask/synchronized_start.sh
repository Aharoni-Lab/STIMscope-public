#!/bin/bash

# Synchronized Mask Projection System Startup
# Ensures proper timing to avoid garbage mask ID mapping

set -e

echo "🚀 Starting Synchronized Mask Projection System"

# Clean up any existing files
echo "🧹 Cleaning up previous session files..."
rm -f mask_map.csv sent_masks.csv final_mask_to_frame.csv

# Step 1: Start the projector (it will wait for mask data)
echo "📽️  Starting projector system..."
cd /media/aharonilabjetson2/NVMe/projects/STIMViewerV2/MyUART/ZMQ_sender_mask
./projector --latency-frames=4 --visible-id=1 &
PROJECTOR_PID=$!
echo "   Projector PID: $PROJECTOR_PID"

# Step 2: Wait a moment for projector to initialize
echo "⏳ Waiting for projector to initialize..."
sleep 2

# Step 3: Start mask sender
echo "🎭 Starting mask sender..."
cd /home/aharonilabjetson2/Desktop/MyScripts/MyUART/ZMQ_sender_mask
python3 zmq_mask_sender.py &
SENDER_PID=$!
echo "   Sender PID: $SENDER_PID"

# Step 4: Wait for user to start recording and press key
echo ""
echo "🎥 NOW START YOUR CAMERA RECORDING"
echo "   Press ENTER when recording has started..."
read -p "   " 

# Step 5: Let system run
echo "✅ System running in synchronized mode"
echo "   - Projector: PID $PROJECTOR_PID"
echo "   - Sender: PID $SENDER_PID"
echo "   - CSV mapping will be written to mask_map.csv"
echo ""
echo "Press Ctrl+C to stop all processes"

# Wait for interrupt and clean shutdown
trap 'echo "🛑 Shutting down..."; kill $PROJECTOR_PID $SENDER_PID 2>/dev/null; exit 0' INT

# Keep script running
while true; do
    if ! kill -0 $PROJECTOR_PID 2>/dev/null || ! kill -0 $SENDER_PID 2>/dev/null; then
        echo "⚠️  One of the processes died"
        break
    fi
    sleep 1
done

echo "✅ Shutdown complete"