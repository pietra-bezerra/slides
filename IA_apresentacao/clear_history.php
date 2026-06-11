<?php
$dir = '../captures/';
$history_file = '../history.json';

// Apagar arquivos na pasta captures
if (is_dir($dir)) {
    $files = glob($dir . '*');
    foreach ($files as $file) {
        if (is_file($file)) {
            unlink($file);
        }
    }
}

// Resetar o arquivo de histórico
file_put_contents($history_file, json_encode([]));

header('Location: index.html');
exit;
?>
