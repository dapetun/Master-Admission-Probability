// Вставьте ВЕСЬ текст в Console (F12) на странице с кнопками (уже залогинены).
// Сработает в Chrome / Edge. Один раз разрешите доступ к папке — дальше без окон «Сохранить».

(async () => {
  const links = [...document.querySelectorAll('a.ms-dl-chip--budget[href*="_Budget.xlsx"]')];
  console.log('Найдено файлов: ' + links.length);
  if (!links.length) {
    console.error('Ссылки не найдены. Проверьте селектор.');
    return;
  }

  // Один диалог: выберите папку (например Downloads или data/raw)
  const dir = await window.showDirectoryPicker({ mode: 'readwrite' });

  for (let i = 0; i < links.length; i++) {
    const a = links[i];
    const url = a.href;
    const name = decodeURIComponent(url.split('/').pop());
    try {
      const res = await fetch(url, { credentials: 'include' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const blob = await res.blob();
      const fileHandle = await dir.getFileHandle(name, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      console.log('OK ' + (i + 1) + '/' + links.length + ': ' + name);
    } catch (err) {
      console.error('FAIL ' + (i + 1) + '/' + links.length + ': ' + name, err);
    }
    // небольшая пауза, чтобы не долбить сервер
    await new Promise((r) => setTimeout(r, 300));
  }
  console.log('Готово.');
})();
