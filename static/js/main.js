document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const voiceBtn = document.getElementById('voice-btn');

    // 全局语音控制状态
    const speechControl = {
        currentUtterance: null,
        currentButton: null,
        isPaused: false,
        stopSpeech() {
            if (this.currentUtterance) {
                speechSynthesis.cancel();
                this.currentUtterance = null;
                this.isPaused = false;
                if (this.currentButton) {
                    this.currentButton.classList.remove('playing', 'paused');
                    this.currentButton.title = '播放语音';
                }
            }
        }
    };

    // 语音识别初始化
    let recognition;
    let isListening = false;
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'zh-CN';
        recognition.interimResults = false;

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
        };

        recognition.onerror = (event) => {
            console.error('语音识别错误:', event.error);
        };

        recognition.onend = () => {
            isListening = false;
            voiceBtn.classList.remove('listening');
        };
    } else {
        voiceBtn.disabled = true;
        voiceBtn.title = '浏览器不支持语音识别';
    }

    // 语音输入控制
    voiceBtn.addEventListener('click', () => {
        if (!isListening) {
            speechControl.stopSpeech();
            recognition.start();
            isListening = true;
            voiceBtn.classList.add('listening');
        } else {
            recognition.stop();
            isListening = false;
            voiceBtn.classList.remove('listening');
        }
    });

    // 创建智能播放按钮
    function createPlayButton(text) {
        const button = document.createElement('i');
        button.className = 'fas fa-volume-up play-button';
        button.title = '播放语音';

        button.addEventListener('click', () => {
            // 停止其他语音
            if (speechControl.currentButton !== button) {
                speechControl.stopSpeech();
            }

            if (!speechControl.currentUtterance) {
                // 新建播放
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'zh-CN';
                utterance.rate = 0.9;

                utterance.onstart = () => {
                    button.classList.add('playing');
                    button.title = '正在播放 - 点击暂停';
                    speechControl.currentUtterance = utterance;
                    speechControl.currentButton = button;
                    speechControl.isPaused = false;
                };

                utterance.onend = () => {
                    button.classList.remove('playing', 'paused');
                    speechControl.currentUtterance = null;
                    speechControl.currentButton = null;
                };

                utterance.onpause = () => {
                    button.classList.replace('playing', 'paused');
                    button.title = '已暂停 - 点击继续';
                    speechControl.isPaused = true;
                };

                speechSynthesis.speak(utterance);
            } else if (speechControl.isPaused) {
                // 恢复播放
                speechSynthesis.resume();
                button.classList.replace('paused', 'playing');
                button.title = '正在播放 - 点击暂停';
                speechControl.isPaused = false;
            } else {
                // 暂停播放
                speechSynthesis.pause();
                button.classList.replace('playing', 'paused');
                button.title = '已暂停 - 点击继续';
                speechControl.isPaused = true;
            }
        });

        return button;
    }

    // 添加消息到聊天框
    function addMessage(role, content, references = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        // 消息内容容器
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'content-wrapper';

        // 文本内容
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.textContent = content;

        // 添加播放按钮（仅助理消息）
        if (role === 'assistant') {
            const playButton = createPlayButton(content);
            contentWrapper.appendChild(playButton);
        }





        // 参考资料

        if (references.length > 0) {
            const refDiv = document.createElement('div');
            refDiv.className = 'references';
            refDiv.innerHTML = `
                <div class="ref-header">参考资料：</div>
                ${references.map((ref, i) => `
                    <div class="ref-item">
                        <span class="ref-index">${i+1}.</span>
                        ${ref}
                    </div>
                `).join('')}
            `;
            messageDiv.appendChild(refDiv);
        }

        contentWrapper.appendChild(contentDiv);
        messageDiv.appendChild(contentWrapper);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // 自动调整消息宽度
        setTimeout(() => {
            const contentWidth = contentDiv.scrollWidth;
            const maxWidth = chatBox.offsetWidth * 0.85;
            messageDiv.style.maxWidth = `${Math.min(contentWidth + 80, maxWidth)}px`;
        }, 10);
    }

    // 初始化欢迎信息
    document.querySelector('.welcome-message').style.display = 'block';

    // 处理消息发送
    async function handleSend() {
        const question = userInput.value.trim();
        if (!question) return;

        userInput.value = '';
        speechControl.stopSpeech(); // 发送时停止语音
        addMessage('user', question);

        // 加载状态
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading';
        loadingDiv.innerHTML = '<div class="content">正在思考中...</div>';
        chatBox.appendChild(loadingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: question })
            });

            const data = await response.json();
            chatBox.removeChild(loadingDiv);

            if (data.status === 'success') {
                addMessage('assistant', data.response, data.references);
            } else {
                addMessage('assistant', `错误：${data.message}`);
            }
        } catch (error) {
            chatBox.removeChild(loadingDiv);
            addMessage('assistant', `请求失败：${error.message}`);
        }
    }

    // 事件监听
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // 页面隐藏时暂停语音
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && speechControl.currentUtterance) {
            speechSynthesis.pause();
            speechControl.isPaused = true;
            speechControl.currentButton?.classList.replace('playing', 'paused');
        }
    });
});