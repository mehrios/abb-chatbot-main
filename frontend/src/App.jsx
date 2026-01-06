import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Send, Mic, MicOff, LogOut, BarChart3, MessageSquare, User, Star, Clock, DollarSign, TrendingUp } from 'lucide-react';

// API Configuration
const API_BASE_URL = 'http://localhost:8100';

// Language configurations
const LANGUAGES = {
  az: {
    welcome: 'Xoş gəlmisiniz! ABB Bank çatbotuna daxil olun və ya qeydiyyatdan keçin',
    login: 'Daxil ol',
    register: 'Qeydiyyat',
    username: 'İstifadəçi adı',
    password: 'Şifrə',
    selectLanguage: 'Ünsiyyət dilini seçin',
    greeting: 'Salam! ABB Bank çatbotuna xoş gəldiniz. Sizə necə kömək edə bilərəm?',
    typeMessage: 'Mesajınızı yazın...',
    rateResponse: 'Cavabı qiymətləndirin',
    submit: 'Göndər',
    logout: 'Çıxış',
    dashboard: 'İdarə paneli',
    chat: 'Söhbət',
    resetPassword: 'Şifrəni sıfırla',
    newPassword: 'Yeni şifrə',
    recording: 'Qeyd edilir...',
    stopRecording: 'Dayandır',
    voiceNotSupported: 'Səs dəstəklənmir',
    processing: 'İşlənir...'
  },
  ru: {
    welcome: 'Добро пожаловать! Войдите в чатбот банка ABB или зарегистрируйтесь',
    login: 'Войти',
    register: 'Регистрация',
    username: 'Имя пользователя',
    password: 'Пароль',
    selectLanguage: 'Выберите язык общения',
    greeting: 'Здравствуйте! Добро пожаловать в чатбот ABB Bank. Чем могу помочь?',
    typeMessage: 'Введите ваше сообщение...',
    rateResponse: 'Оцените ответ',
    submit: 'Отправить',
    logout: 'Выход',
    dashboard: 'Панель управления',
    chat: 'Чат',
    resetPassword: 'Сбросить пароль',
    newPassword: 'Новый пароль',
    recording: 'Идет запись...',
    stopRecording: 'Остановить',
    voiceNotSupported: 'Голосовой ввод не поддерживается',
    processing: 'Обработка...'
  },
  en: {
    welcome: 'Welcome! Login to ABB Bank chatbot or register',
    login: 'Login',
    register: 'Register',
    username: 'Username',
    password: 'Password',
    selectLanguage: 'Select communication language',
    greeting: 'Hello! Welcome to ABB Bank chatbot. How can I help you?',
    typeMessage: 'Type your message...',
    rateResponse: 'Rate the response',
    submit: 'Submit',
    logout: 'Logout',
    dashboard: 'Dashboard',
    chat: 'Chat',
    resetPassword: 'Reset password',
    newPassword: 'New password',
    recording: 'Recording...',
    stopRecording: 'Stop',
    voiceNotSupported: 'Voice input not supported',
    processing: 'Processing...'
  }
};

// Auth Component
const AuthScreen = ({ onLogin, onRegister }) => {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [language, setLanguage] = useState('az');
  const [error, setError] = useState('');

  const t = LANGUAGES[language];

  const handleSubmit = async () => {
    setError('');

    try {
      if (mode === 'login') {
        const response = await fetch(`${API_BASE_URL}/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        });

        if (!response.ok) {
          setError('Invalid username or password');
          return;
        }

        const user = await response.json();
        onLogin(user, language);
      } else if (mode === 'register') {
        const response = await fetch(`${API_BASE_URL}/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password, preferred_language: language })
        });

        if (!response.ok) {
          setError('Username already exists');
          return;
        }

        const user = await response.json();
        onRegister(user, language);
      } else if (mode === 'reset') {
        const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, new_password: newPassword })
        });

        if (!response.ok) {
          setError('User not found');
          return;
        }

        setMode('login');
        setPassword(newPassword);
      }
    } catch (err) {
      setError('Connection error');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-indigo-600 mb-2">ABB Bank</h1>
          <p className="text-gray-600">{t.welcome}</p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t.username}
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {mode === 'reset' ? t.newPassword : t.password}
            </label>
            <input
              type="password"
              value={mode === 'reset' ? newPassword : password}
              onChange={(e) => mode === 'reset' ? setNewPassword(e.target.value) : setPassword(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          {mode !== 'reset' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t.selectLanguage}
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setLanguage('az')}
                  className={`py-3 px-4 rounded-lg font-medium transition ${
                    language === 'az'
                      ? 'bg-indigo-600 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  AZ
                </button>
                <button
                  onClick={() => setLanguage('ru')}
                  className={`py-3 px-4 rounded-lg font-medium transition ${
                    language === 'ru'
                      ? 'bg-indigo-600 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  RU
                </button>
                <button
                  onClick={() => setLanguage('en')}
                  className={`py-3 px-4 rounded-lg font-medium transition ${
                    language === 'en'
                      ? 'bg-indigo-600 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  EN
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm">
              {error}
            </div>
          )}

          <button
            onClick={handleSubmit}
            className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
          >
            {mode === 'login' ? t.login : mode === 'register' ? t.register : t.resetPassword}
          </button>

          <div className="flex justify-between text-sm">
            {mode === 'login' ? (
              <>
                <button onClick={() => setMode('register')} className="text-indigo-600 hover:underline">
                  {t.register}
                </button>
                <button onClick={() => setMode('reset')} className="text-indigo-600 hover:underline">
                  {t.resetPassword}
                </button>
              </>
            ) : (
              <button onClick={() => setMode('login')} className="text-indigo-600 hover:underline">
                {t.login}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Chat Component with Voice Recording
const ChatScreen = ({ user, language, onLogout, onNavigate }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showRating, setShowRating] = useState(false);
  const [currentRating, setCurrentRating] = useState(0);
  const [lastQueryLogId, setLastQueryLogId] = useState(null);
  const [chatId] = useState(`chat_${user.id}_${Date.now()}`);
  const messagesEndRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);

  const t = LANGUAGES[language];

  useEffect(() => {
    const initChat = async () => {
      try {
        await fetch(`${API_BASE_URL}/chats`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: chatId,
            user_id: user.id,
            header: 'New Chat',
            language: language
          })
        });

        setMessages([{
          text: t.greeting,
          sender: 'bot',
          timestamp: new Date()
        }]);
      } catch (err) {
        console.error('Chat init error:', err);
      }
    };

    initChat();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          chat_id: chatId,
          user_id: user.id,
          language: language
        })
      });

      const data = await response.json();

      if (data.success) {
        setMessages(prev => [...prev, {
          text: data.response,
          sender: 'bot',
          timestamp: new Date(),
          cached: data.was_cached
        }]);

        setLastQueryLogId(data.query_log_id);
        setShowRating(true);
      } else {
        setMessages(prev => [...prev, {
          text: data.error || 'Error processing request',
          sender: 'bot',
          timestamp: new Date()
        }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        text: 'Connection error',
        sender: 'bot',
        timestamp: new Date()
      }]);
    }

    setLoading(false);
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert(t.voiceNotSupported);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await sendVoiceMessage(audioBlob);
        
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Recording error:', err);
      alert('Микрофон недоступен');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    }
  };

  const sendVoiceMessage = async (audioBlob) => {
    setLoading(true);

    const userMessage = {
      text: `🎤 ${t.processing}`,
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('chat_id', chatId);
      formData.append('user_id', user.id);
      formData.append('language', language);

      const response = await fetch(`${API_BASE_URL}/query/voice`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            text: `🎤 "${data.transcription}"`,
            sender: 'user',
            timestamp: new Date()
          };
          return newMessages;
        });

        setMessages(prev => [...prev, {
          text: data.response,
          sender: 'bot',
          timestamp: new Date(),
          cached: data.was_cached
        }]);

        setLastQueryLogId(data.query_log_id);
        setShowRating(true);
      } else {
        setMessages(prev => [...prev, {
          text: data.error || 'Voice processing error',
          sender: 'bot',
          timestamp: new Date()
        }]);
      }
    } catch (err) {
      console.error('Voice send error:', err);
      setMessages(prev => [...prev, {
        text: 'Connection error',
        sender: 'bot',
        timestamp: new Date()
      }]);
    }

    setLoading(false);
  };

  const handleRating = async (rating) => {
    setCurrentRating(rating);

    try {
      await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.id,
          chat_id: chatId,
          query_log_id: lastQueryLogId,
          rating: rating
        })
      });

      setShowRating(false);
      setCurrentRating(0);
    } catch (err) {
      console.error('Feedback error:', err);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <div className="bg-indigo-600 text-white p-4 shadow-lg flex items-center justify-between">
        <div className="flex items-center gap-3">
          <User className="w-6 h-6" />
          <div>
            <div className="font-semibold">{user.username}</div>
            <div className="text-sm text-indigo-200">ABB Bank Chatbot</div>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onNavigate('dashboard')}
            className="p-2 hover:bg-indigo-700 rounded-lg transition"
          >
            <BarChart3 className="w-5 h-5" />
          </button>
          <button
            onClick={onLogout}
            className="p-2 hover:bg-indigo-700 rounded-lg transition"
          >
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                msg.sender === 'user'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-gray-800 shadow-md'
              }`}
            >
              <div className="text-sm">{msg.text}</div>
              {msg.cached && (
                <div className="text-xs mt-1 opacity-75">⚡ Cached</div>
              )}
              <div className="text-xs mt-1 opacity-75">
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white px-4 py-3 rounded-2xl shadow-md">
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {showRating && (
        <div className="bg-yellow-50 border-t border-yellow-200 p-4">
          <div className="text-center mb-2 text-sm font-medium text-gray-700">
            {t.rateResponse}
          </div>
          <div className="flex justify-center gap-2">
            {[1, 2, 3, 4, 5].map((rating) => (
              <button
                key={rating}
                onClick={() => handleRating(rating)}
                className="text-2xl hover:scale-110 transition"
              >
                <Star
                  className={`w-8 h-8 ${
                    rating <= currentRating ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'
                  }`}
                />
              </button>
            ))}
          </div>
        </div>
      )}

      {isRecording && (
        <div className="bg-red-50 border-t border-red-200 p-3">
          <div className="flex items-center justify-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-red-600 font-medium">{t.recording}</span>
            </div>
            <span className="text-red-600 font-mono">{recordingTime}s</span>
          </div>
        </div>
      )}

      <div className="bg-white border-t p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder={t.typeMessage}
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            disabled={loading || isRecording}
          />
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`p-3 rounded-lg transition ${
              isRecording 
                ? 'bg-red-500 hover:bg-red-600 text-white' 
                : 'bg-gray-200 hover:bg-gray-300 text-gray-600'
            }`}
            disabled={loading}
          >
            {isRecording ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
          </button>
          <button
            onClick={handleSend}
            className="bg-indigo-600 text-white p-3 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
            disabled={loading || !input.trim() || isRecording}
          >
            <Send className="w-6 h-6" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Dashboard Component
const DashboardScreen = ({ user, language, onNavigate }) => {
  const [filters, setFilters] = useState({
    start_date: '',
    end_date: '',
    username: '',
    min_rating: '',
    max_rating: ''
  });
  const [costData, setCostData] = useState([]);
  const [llmTimeData, setLlmTimeData] = useState([]);
  const [retrieverTimeData, setRetrieverTimeData] = useState([]);
  const [ratingsData, setRatingsData] = useState([]);
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalTokens: 0,
    avgTokens: 0,
    totalMessages: 0  // Добавлено
  });
  const [loading, setLoading] = useState(false);

  const t = LANGUAGES[language];

  useEffect(() => {
    loadDashboardData();
  }, []);

  const buildFilterBody = () => {
    const filterBody = {};
    
    if (filters.start_date) {
      filterBody.start_date = filters.start_date;
    }
    if (filters.end_date) {
      filterBody.end_date = filters.end_date;
    }
    if (filters.username) {
      filterBody.username = filters.username;
    }
    if (filters.min_rating) {
      filterBody.min_rating = parseInt(filters.min_rating);
    }
    if (filters.max_rating) {
      filterBody.max_rating = parseInt(filters.max_rating);
    }
    
    return filterBody;
  };

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const filterBody = buildFilterBody();

      // Cost per day
      const costRes = await fetch(`${API_BASE_URL}/analytics/cost/per-day`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      
      if (costRes.ok) {
        const costJson = await costRes.json();
        setCostData(Object.entries(costJson).map(([day, data]) => ({
          day,
          cost: data.cost || 0,
          count: data.count || 0
        })));
      } else {
        console.error('Cost data error:', await costRes.text());
      }

      // LLM time
      const llmRes = await fetch(`${API_BASE_URL}/analytics/llm-time/per-day`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      
      if (llmRes.ok) {
        const llmJson = await llmRes.json();
        setLlmTimeData(Object.entries(llmJson).map(([day, data]) => ({
          day,
          time: data.avg_time || 0,
          count: data.count || 0
        })));
      }

      // Retriever time
      const retRes = await fetch(`${API_BASE_URL}/analytics/retriever-time/per-day`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      
      if (retRes.ok) {
        const retJson = await retRes.json();
        setRetrieverTimeData(Object.entries(retJson).map(([day, data]) => ({
          day,
          time: data.avg_time || 0,
          count: data.count || 0
        })));
      }

      // Ratings
      const ratRes = await fetch(`${API_BASE_URL}/analytics/ratings/per-day`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      
      if (ratRes.ok) {
        const ratJson = await ratRes.json();
        setRatingsData(Object.entries(ratJson).map(([day, data]) => ({
          day,
          rating: data.avg_rating || 0,
          count: data.count || 0
        })));
      }

      // Stats
      const usersRes = await fetch(`${API_BASE_URL}/analytics/users/count`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      const usersData = await usersRes.json();

      const tokensRes = await fetch(`${API_BASE_URL}/analytics/tokens/total`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      const tokensData = await tokensRes.json();

      const avgRes = await fetch(`${API_BASE_URL}/analytics/tokens/average`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      const avgData = await avgRes.json();

      // Добавлено: Получение количества сообщений
      const messagesRes = await fetch(`${API_BASE_URL}/analytics/messages/count`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filterBody)
      });
      const messagesData = await messagesRes.json();

      setStats({
        totalUsers: usersData.count || 0,
        totalTokens: tokensData.total || 0,
        avgTokens: avgData.average || 0,
        totalMessages: messagesData.count || 0  // Добавлено
      });
    } catch (err) {
      console.error('Dashboard loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleApplyFilters = () => {
    loadDashboardData();
  };

  const handleResetFilters = () => {
    setFilters({
      start_date: '',
      end_date: '',
      username: '',
      min_rating: '',
      max_rating: ''
    });
    setTimeout(() => {
      loadDashboardData();
    }, 100);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-indigo-600 text-white p-4 shadow-lg flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-6 h-6" />
          <div className="font-semibold text-lg">{t.dashboard}</div>
        </div>
        <button
          onClick={() => onNavigate('chat')}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-700 hover:bg-indigo-800 rounded-lg transition"
        >
          <MessageSquare className="w-5 h-5" />
          {t.chat}
        </button>
      </div>

      <div className="p-6 space-y-6">
        {/* Filters Section */}
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            Filters
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Start Date
              </label>
              <input
                type="date"
                value={filters.start_date}
                onChange={(e) => handleFilterChange('start_date', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                End Date
              </label>
              <input
                type="date"
                value={filters.end_date}
                onChange={(e) => handleFilterChange('end_date', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Username
              </label>
              <input
                type="text"
                value={filters.username}
                onChange={(e) => handleFilterChange('username', e.target.value)}
                placeholder="Enter username"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Rating
              </label>
              <select
                value={filters.min_rating}
                onChange={(e) => handleFilterChange('min_rating', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              >
                <option value="">All</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Rating
              </label>
              <select
                value={filters.max_rating}
                onChange={(e) => handleFilterChange('max_rating', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              >
                <option value="">All</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
              </select>
            </div>
          </div>

          <div className="flex gap-3 mt-4">
            <button
              onClick={handleApplyFilters}
              disabled={loading}
              className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition font-medium disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Apply Filters'}
            </button>
            <button
              onClick={handleResetFilters}
              disabled={loading}
              className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition font-medium disabled:opacity-50"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Stats Cards - ОБНОВЛЕНО: 4 карточки в ряд */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-gray-500 text-sm">Total Users</div>
                <div className="text-2xl font-bold text-indigo-600">{stats.totalUsers}</div>
              </div>
              <User className="w-12 h-12 text-indigo-200" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-gray-500 text-sm">Total Messages</div>
                <div className="text-2xl font-bold text-blue-600">{stats.totalMessages.toLocaleString()}</div>
              </div>
              <MessageSquare className="w-12 h-12 text-blue-200" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-gray-500 text-sm">Total Tokens</div>
                <div className="text-2xl font-bold text-green-600">{stats.totalTokens.toLocaleString()}</div>
              </div>
              <TrendingUp className="w-12 h-12 text-green-200" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-gray-500 text-sm">Avg Tokens</div>
                <div className="text-2xl font-bold text-purple-600">{stats.avgTokens.toFixed(0)}</div>
              </div>
              <BarChart3 className="w-12 h-12 text-purple-200" />
            </div>
          </div>
        </div>

        {/* Cost Chart */}
        <div className="bg-white p-6 rounded-xl shadow-md">
          <div className="flex items-center gap-2 mb-4">
            <DollarSign className="w-5 h-5 text-indigo-600" />
            <h3 className="text-lg font-semibold">Cost per Day</h3>
          </div>
          {costData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={costData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="cost" fill="#6366f1" name="Cost (USD)" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-400">
              No data available
            </div>
          )}
        </div>

        {/* Time Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-indigo-600" />
              <h3 className="text-lg font-semibold">LLM Response Time</h3>
            </div>
            {llmTimeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={llmTimeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="time" stroke="#10b981" name="Time (s)" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-400">
                No data available
              </div>
            )}
          </div>

          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-indigo-600" />
              <h3 className="text-lg font-semibold">Retriever Response Time</h3>
            </div>
            {retrieverTimeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={retrieverTimeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="time" stroke="#f59e0b" name="Time (s)" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-400">
                No data available
              </div>
            )}
          </div>
        </div>

        {/* Rating Chart */}
        <div className="bg-white p-6 rounded-xl shadow-md">
          <div className="flex items-center gap-2 mb-4">
            <Star className="w-5 h-5 text-indigo-600" />
            <h3 className="text-lg font-semibold">Average Rating per Day</h3>
          </div>
          {ratingsData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ratingsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis domain={[0, 5]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="rating" stroke="#ec4899" name="Rating" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-400">
              No data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Main App
export default function App() {
  const [currentScreen, setCurrentScreen] = useState('auth');
  const [user, setUser] = useState(null);
  const [language, setLanguage] = useState('az');

  const handleLogin = (userData, lang) => {
    setUser(userData);
    setLanguage(lang);
    setCurrentScreen('chat');
  };

  const handleRegister = (userData, lang) => {
    setUser(userData);
    setLanguage(lang);
    setCurrentScreen('chat');
  };

  const handleLogout = () => {
    setUser(null);
    setCurrentScreen('auth');
  };

  const handleNavigate = (screen) => {
    setCurrentScreen(screen);
  };

  if (currentScreen === 'auth') {
    return <AuthScreen onLogin={handleLogin} onRegister={handleRegister} />;
  }

  if (currentScreen === 'chat') {
    return (
      <ChatScreen
        user={user}
        language={language}
        onLogout={handleLogout}
        onNavigate={handleNavigate}
      />
    );
  }

  if (currentScreen === 'dashboard') {
    return (
      <DashboardScreen
        user={user}
        language={language}
        onNavigate={handleNavigate}
      />
    );
  }

  return null;
}