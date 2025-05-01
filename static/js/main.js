// 主题切换
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeToggle');
    const html = document.documentElement;
    
    // 检查本地存储中的主题设置
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        html.classList.toggle('dark', savedTheme === 'dark');
        html.setAttribute('data-theme', savedTheme);
    }
    
    themeToggle.addEventListener('click', () => {
        html.classList.toggle('dark');
        const isDark = html.classList.contains('dark');
        html.setAttribute('data-theme', isDark ? 'dark' : 'light');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
});

// 文件拖放功能
document.addEventListener('DOMContentLoaded', () => {
    const uploadContainer = document.querySelector('.upload-container');
    const fileInput = document.getElementById('audioFile');
    
    uploadContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadContainer.classList.add('border-green-500');
    });
    
    uploadContainer.addEventListener('dragleave', () => {
        uploadContainer.classList.remove('border-green-500');
    });
    
    uploadContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadContainer.classList.remove('border-green-500');
        
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
});

// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
}); 