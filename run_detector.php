<?php
// Execute python detector and log output
$cmd = '"C:\Program Files\Python314\python.exe" IA_apresentacao\detector.py > detector_log.txt 2>&1';
pclose(popen('start "" ' . $cmd, "r"));
header('Content-Type: application/json');
echo json_encode(["status" => "success"]);
?>
