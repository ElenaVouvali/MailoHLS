# GAN FiLM: Vitis HLS 2021.1 synthesis of 18 requests

Area (%) = 100 × mean(BRAM18K/capacity, DSP/capacity, FF/capacity, LUT/capacity).
Speedup = pragma-free baseline worst-case cycles / predicted worst-case cycles at the same part and requested clock.
Both use the same application headers and synthesis settings. Budget-only duplicates share physical measurements.
Timing is an HLS estimate, not post-route timing; no functional simulation was run.

|Device|Clock ns|Budget %|Mean utilization %|Speedup|Baseline cycles|Predicted cycles|Budget met|Estimated clock ns|HLS timing met|
|---|---|---|---|---|---|---|---|---|---|
|ZCU104|3.33|20|6.3547|34.094×|158195|4640|True|2.404|True|
|ZCU104|3.33|50|5.8170|24.858×|158195|6364|True|2.404|True|
|ZCU104|3.33|100|5.8170|24.858×|158195|6364|True|2.404|True|
|ZCU104|5.0|20|5.7650|23.543×|145259|6170|True|3.557|True|
|ZCU104|5.0|50|5.7650|23.543×|145259|6170|True|3.557|True|
|ZCU104|5.0|100|5.7716|23.566×|145259|6164|True|3.557|True|
|ZCU104|10.0|20|5.8106|22.985×|132714|5774|True|7.095|True|
|ZCU104|10.0|50|6.3293|32.883×|132714|4036|True|7.095|True|
|ZCU104|10.0|100|5.8167|23.009×|132714|5768|True|7.095|True|
|U200|3.33|20|1.0129|25.592×|157803|6166|True|2.425|True|
|U200|3.33|50|1.0129|25.592×|157803|6166|True|2.425|True|
|U200|3.33|100|1.2137|14.620×|157803|10794|True|2.425|True|
|U200|5.0|20|1.0017|24.323×|145258|5972|True|3.602|True|
|U200|5.0|50|1.0032|24.348×|145258|5966|True|3.602|True|
|U200|5.0|100|1.0032|24.348×|145258|5966|True|3.602|True|
|U200|10.0|20|1.0152|22.941×|132322|5768|True|6.688|True|
|U200|10.0|50|1.0152|22.941×|132322|5768|True|6.688|True|
|U200|10.0|100|1.0152|22.941×|132322|5768|True|6.688|True|
