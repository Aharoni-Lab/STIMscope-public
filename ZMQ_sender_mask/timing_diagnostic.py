#!/usr/bin/env python3
"""
Timing Diagnostic Script for Mask Projection System
Analyzes CSV files and provides detailed timing validation
"""

import csv
import sys
from collections import Counter

def analyze_mask_map(csv_path):
    """Analyze the mask_map.csv for timing issues"""
    print(f"\n🔍 Analyzing {csv_path}")
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            entries = list(reader)
        
        if not entries:
            print("❌ Empty CSV file")
            return
            
        print(f"📊 Total entries: {len(entries)}")
        
        # Parse entries
        valid_entries = []
        invalid_entries = []
        
        for i, line in enumerate(entries):
            if len(line) >= 2:
                try:
                    mask_id = int(line[0])
                    frame_num = int(line[1])
                    
                    if mask_id < 0 or mask_id > 1000000:
                        invalid_entries.append((i+1, mask_id, frame_num))
                    else:
                        valid_entries.append((mask_id, frame_num))
                        
                except ValueError:
                    invalid_entries.append((i+1, line[0], line[1]))
        
        print(f"✅ Valid entries: {len(valid_entries)}")
        print(f"❌ Invalid entries: {len(invalid_entries)}")
        
        if invalid_entries:
            print("\n⚠️  Invalid entries found:")
            for line_num, mask_id, frame_num in invalid_entries[:10]:  # Show first 10
                print(f"   Line {line_num}: mask_id={mask_id}, frame={frame_num}")
            if len(invalid_entries) > 10:
                print(f"   ... and {len(invalid_entries)-10} more")
        
        if valid_entries:
            # Analyze valid entries
            mask_ids = [entry[0] for entry in valid_entries]
            frame_nums = [entry[1] for entry in valid_entries]
            
            print(f"\n📈 Valid entry analysis:")
            print(f"   Mask ID range: {min(mask_ids)} to {max(mask_ids)}")
            print(f"   Frame range: {min(frame_nums)} to {max(frame_nums)}")
            print(f"   First valid mapping: mask_id={mask_ids[0]} -> frame={frame_nums[0]}")
            
            # Check for frame continuity
            frame_gaps = []
            for i in range(1, len(frame_nums)):
                gap = frame_nums[i] - frame_nums[i-1]
                if gap > 2:  # Allow for small gaps due to 60Hz->30Hz conversion
                    frame_gaps.append((frame_nums[i-1], frame_nums[i], gap))
            
            if frame_gaps:
                print(f"   ⚠️  {len(frame_gaps)} large frame gaps found:")
                for prev_frame, next_frame, gap in frame_gaps[:5]:
                    print(f"      {prev_frame} -> {next_frame} (gap: {gap})")
            else:
                print("   ✅ No significant frame gaps detected")
                
            # Check mask ID progression
            mask_gaps = []
            for i in range(1, len(mask_ids)):
                gap = mask_ids[i] - mask_ids[i-1]
                if gap < 0 or gap > 10:  # Expect mostly sequential with some gaps
                    mask_gaps.append((mask_ids[i-1], mask_ids[i], gap))
            
            if mask_gaps:
                print(f"   ⚠️  {len(mask_gaps)} irregular mask ID progressions:")
                for prev_id, next_id, gap in mask_gaps[:5]:
                    print(f"      {prev_id} -> {next_id} (gap: {gap})")
    
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
    except Exception as e:
        print(f"❌ Error analyzing {csv_path}: {e}")

def compare_with_sent_masks(mask_map_path, sent_masks_path):
    """Compare mask_map.csv with sent_masks.csv"""
    print(f"\n🔄 Comparing mappings with sent masks...")
    
    try:
        # Load sent masks
        sent_masks = set()
        with open(sent_masks_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mask_id = int(row['mask_id'])
                if row['status'] == 'sent':
                    sent_masks.add(mask_id)
        
        print(f"📤 Sent masks: {len(sent_masks)}")
        print(f"   Range: {min(sent_masks)} to {max(sent_masks)}")
        
        # Load mapped masks
        mapped_masks = set()
        with open(mask_map_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 2:
                    try:
                        mask_id = int(line[0])
                        if 0 <= mask_id <= 1000000:  # Valid range
                            mapped_masks.add(mask_id)
                    except ValueError:
                        continue
        
        print(f"🎯 Mapped masks: {len(mapped_masks)}")
        if mapped_masks:
            print(f"   Range: {min(mapped_masks)} to {max(mapped_masks)}")
        
        # Compare
        sent_not_mapped = sent_masks - mapped_masks
        mapped_not_sent = mapped_masks - sent_masks
        
        print(f"\n📊 Comparison:")
        print(f"   Sent but not mapped: {len(sent_not_mapped)}")
        print(f"   Mapped but not sent: {len(mapped_not_sent)}")
        print(f"   Successfully mapped: {len(sent_masks & mapped_masks)}")
        
        if sent_not_mapped:
            missing_sample = list(sorted(sent_not_mapped))[:10]
            print(f"   Missing examples: {missing_sample}")
            
    except FileNotFoundError as e:
        print(f"❌ Comparison file not found: {e}")
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

def main():
    """Main diagnostic function"""
    print("🏥 Mask Projection Timing Diagnostic")
    print("=====================================")
    
    # Default paths
    mask_map_path = sys.argv[1] if len(sys.argv) > 1 else "mask_map.csv"
    sent_masks_path = sys.argv[2] if len(sys.argv) > 2 else "sent_masks.csv"
    
    # Analyze mask mapping
    analyze_mask_map(mask_map_path)
    
    # Compare with sent masks if available
    try:
        compare_with_sent_masks(mask_map_path, sent_masks_path)
    except:
        print("\n⚠️  Could not compare with sent_masks.csv")
    
    print("\n✅ Diagnostic complete!")
    print("\n💡 Recommendations:")
    print("   1. Use the synchronized_start.sh script to avoid garbage values")
    print("   2. Ensure LATENCY_FRAMES=4 for 60Hz->30Hz conversion")
    print("   3. Check that mask_id values are reasonable (1-1000000 range)")
    print("   4. Verify frame progression matches camera recording")

if __name__ == "__main__":
    main()