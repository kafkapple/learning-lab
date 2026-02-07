/**
 * ADHD Learning System - Dashboard JavaScript
 *
 * 인터랙티브 대시보드 기능:
 * - 실시간 데이터 로딩
 * - GitHub 스타일 히트맵 캘린더
 * - 포모도로 타이머
 * - 주간 차트
 */

// ===== 전역 상태 =====
const state = {
    user: 'default_user',
    dashboard: null,
    heatmap: null,
    timer: {
        isRunning: false,
        isPaused: false,
        timeLeft: 25 * 60, // 초
        workDuration: 25 * 60,
        breakDuration: 5 * 60,
        longBreakDuration: 15 * 60,
        sessionCount: 0,
        isBreak: false
    }
};

// ===== 초기화 =====
document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
    initPomodoro();
    initTooltip();
});

async function initDashboard() {
    // 날짜 표시
    document.getElementById('currentDate').textContent = formatDate(new Date());

    // 데이터 로딩
    await Promise.all([
        loadDashboardData(),
        loadHeatmapData(),
        loadQuests()
    ]);

    // 주간 차트 초기화
    initWeeklyChart();
}

// ===== API 호출 =====
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`/api${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return { success: false, error: error.message };
    }
}

// ===== 대시보드 데이터 =====
async function loadDashboardData() {
    const result = await fetchAPI(`/dashboard?user_id=${state.user}`);

    if (result.success) {
        state.dashboard = result.data;
        updateDashboardUI(result.data);
    }
}

function updateDashboardUI(data) {
    const { gamification, statistics } = data;

    // 레벨 & XP
    document.getElementById('userLevel').textContent = gamification.level;
    document.getElementById('currentXP').textContent = gamification.total_xp;

    const xpToNext = gamification.xp_to_next || 100;
    const xpProgress = Math.min((gamification.xp_progress || 0) * 100, 100);
    document.getElementById('nextLevelXP').textContent = gamification.total_xp + xpToNext;
    document.getElementById('xpProgress').style.width = `${xpProgress}%`;

    // 스트릭
    document.getElementById('streakCount').textContent = gamification.current_streak;

    // 오늘 통계
    document.getElementById('todayCards').textContent = gamification.today_cards;
    document.getElementById('todayXP').textContent = gamification.today_xp;
    document.getElementById('dueCards').textContent = statistics.due_today;

    // 복습 버튼 뱃지
    document.getElementById('reviewBadge').textContent = statistics.due_today;

    // 동기 부여 메시지
    document.getElementById('motivationText').textContent = gamification.motivational_message;

    // 히트맵 통계
    document.getElementById('longestStreak').textContent = gamification.longest_streak;
    document.getElementById('totalCards').textContent = statistics.total_cards;

    // 뱃지
    updateBadges(gamification.badges_earned);
}

function updateBadges(badgesEarned) {
    const badgesGrid = document.getElementById('badgesGrid');
    const allBadges = [
        { id: 'streak_3', icon: '🔥', name: '3일 연속' },
        { id: 'streak_7', icon: '⚔️', name: '일주일 전사' },
        { id: 'streak_30', icon: '👑', name: '월간 마스터' },
        { id: 'first_100', icon: '📚', name: '첫 100장' },
        { id: 'perfect_day', icon: '⭐', name: '완벽한 하루' },
        { id: 'explorer_5', icon: '🗺️', name: '탐험가' },
        { id: 'speedster', icon: '⚡', name: '스피드러너' },
        { id: 'night_owl', icon: '🦉', name: '야행성' },
    ];

    badgesGrid.innerHTML = allBadges.map(badge => `
        <div class="badge-item ${badgesEarned > 0 ? '' : 'locked'}">
            <div class="badge-icon">${badge.icon}</div>
            <div class="badge-name">${badge.name}</div>
        </div>
    `).join('');
}

// ===== 히트맵 캘린더 =====
async function loadHeatmapData() {
    const result = await fetchAPI(`/heatmap?user_id=${state.user}&days=365`);

    if (result.success) {
        state.heatmap = result.data;
        renderHeatmap(result.data);
    }
}

function renderHeatmap(data) {
    const grid = document.getElementById('heatmapGrid');
    const monthsContainer = document.getElementById('heatmapMonths');

    // 날짜 계산 (최근 1년)
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - 364);

    // 시작일을 일요일로 조정
    const startDayOfWeek = startDate.getDay();
    startDate.setDate(startDate.getDate() - startDayOfWeek);

    // 그리드 생성
    let cells = [];
    let currentDate = new Date(startDate);
    let totalDays = 0;
    let totalCards = 0;

    // 월 레이블 위치 계산
    const months = [];
    let lastMonth = -1;

    while (currentDate <= today) {
        const dateStr = formatDateISO(currentDate);
        const dayData = data[dateStr] || { count: 0, intensity: 0 };

        // 레벨 계산 (0-4)
        let level = 0;
        if (dayData.count > 0) {
            totalDays++;
            totalCards += dayData.count;
            if (dayData.count >= 50) level = 4;
            else if (dayData.count >= 30) level = 3;
            else if (dayData.count >= 15) level = 2;
            else level = 1;
        }

        // 월 레이블
        const month = currentDate.getMonth();
        if (month !== lastMonth) {
            months.push({
                name: getMonthName(month),
                position: cells.length / 7
            });
            lastMonth = month;
        }

        cells.push({
            date: dateStr,
            level: level,
            count: dayData.count,
            xp: dayData.xp || 0,
            minutes: dayData.minutes || 0
        });

        currentDate.setDate(currentDate.getDate() + 1);
    }

    // 월 레이블 렌더링
    monthsContainer.innerHTML = months.map((m, i) => {
        const nextPos = months[i + 1]?.position || (cells.length / 7);
        const width = ((nextPos - m.position) / (cells.length / 7)) * 100;
        return `<span style="flex: 0 0 ${width}%">${m.name}</span>`;
    }).join('');

    // 셀 렌더링
    grid.innerHTML = cells.map(cell => `
        <div class="heatmap-cell level-${cell.level}"
             data-date="${cell.date}"
             data-count="${cell.count}"
             data-xp="${cell.xp}"
             data-minutes="${cell.minutes}">
        </div>
    `).join('');

    // 통계 업데이트
    document.getElementById('totalDays').textContent = totalDays;
    document.getElementById('totalCards').textContent = totalCards;

    // 셀 이벤트 리스너
    grid.querySelectorAll('.heatmap-cell').forEach(cell => {
        cell.addEventListener('mouseenter', showHeatmapTooltip);
        cell.addEventListener('mouseleave', hideHeatmapTooltip);
        cell.addEventListener('click', showReviewHistory);
    });
}

// ===== 복습 기록 모달 =====
async function showReviewHistory(e) {
    const cell = e.target;
    const date = cell.dataset.date;
    const count = parseInt(cell.dataset.count) || 0;

    if (count === 0) {
        // 복습 기록이 없는 날은 알림만 표시
        showToast('이 날의 복습 기록이 없습니다.');
        return;
    }

    // API 호출하여 복습 기록 가져오기
    const result = await fetchAPI(`/reviews/by-date?date=${date}`);

    if (!result.success) {
        showToast('복습 기록을 불러오는 데 실패했습니다.');
        return;
    }

    const reviews = result.data.reviews;
    renderReviewHistoryModal(date, reviews);
}

function renderReviewHistoryModal(date, reviews) {
    const modal = document.getElementById('reviewHistoryModal');
    const dateEl = document.getElementById('reviewHistoryDate');
    const listEl = document.getElementById('reviewHistoryList');
    const countEl = document.getElementById('reviewHistoryCount');

    // 날짜 표시
    dateEl.textContent = formatDateKorean(date);
    countEl.textContent = `총 ${reviews.length}개의 복습`;

    // 복습 기록 렌더링
    if (reviews.length === 0) {
        listEl.innerHTML = '<div class="empty-state">복습 기록이 없습니다.</div>';
    } else {
        // 지식별로 그룹화
        const groupedByKnowledge = {};
        reviews.forEach(r => {
            const key = r.chunk_id || r.parent_topic || '기타';
            if (!groupedByKnowledge[key]) {
                groupedByKnowledge[key] = {
                    title: r.knowledge_title || r.parent_topic || '알 수 없음',
                    chunk_id: r.chunk_id,
                    reviews: []
                };
            }
            groupedByKnowledge[key].reviews.push(r);
        });

        listEl.innerHTML = Object.entries(groupedByKnowledge).map(([key, group]) => `
            <div class="review-group">
                <div class="review-group-header">
                    <span class="knowledge-title">${group.title}</span>
                    ${group.chunk_id ? `<a href="/library?highlight=${group.chunk_id}" class="knowledge-link">📚 지식 보기</a>` : ''}
                </div>
                <div class="review-items">
                    ${group.reviews.map(r => `
                        <div class="review-item rating-${r.rating}">
                            <div class="review-question">${truncateText(r.question, 60)}</div>
                            <div class="review-meta">
                                <span class="review-time">${formatTime(r.review_time)}</span>
                                <span class="review-rating">${getRatingBadge(r.rating)}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }

    modal.classList.add('active');
}

function closeReviewHistoryModal() {
    document.getElementById('reviewHistoryModal').classList.remove('active');
}

function getRatingBadge(rating) {
    const badges = {
        1: '<span class="badge badge-again">다시</span>',
        2: '<span class="badge badge-hard">어려움</span>',
        3: '<span class="badge badge-good">좋음</span>',
        4: '<span class="badge badge-easy">쉬움</span>'
    };
    return badges[rating] || '';
}

function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function formatTime(datetime) {
    if (!datetime) return '';
    const date = new Date(datetime);
    return date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
}

function showToast(message) {
    // 간단한 토스트 알림
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('visible'), 10);
    setTimeout(() => {
        toast.classList.remove('visible');
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// ===== 퀘스트 =====
async function loadQuests() {
    const result = await fetchAPI(`/quests?user_id=${state.user}`);

    if (result.success) {
        renderQuests(result.data);
    }
}

function renderQuests(quests) {
    const questsList = document.getElementById('questsList');

    questsList.innerHTML = quests.map(quest => `
        <div class="quest-item ${quest.completed ? 'completed' : ''}">
            <div class="quest-checkbox"></div>
            <div class="quest-content">
                <div class="quest-name">${quest.name}</div>
                <div class="quest-desc">${quest.description}</div>
            </div>
            <div class="quest-reward">+${quest.xp_reward}XP</div>
        </div>
    `).join('');
}

// ===== 주간 차트 =====
function initWeeklyChart() {
    const ctx = document.getElementById('weeklyChart');
    if (!ctx) return;

    // 최근 7일 데이터
    const labels = [];
    const cardsData = [];
    const xpData = [];

    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        labels.push(getDayName(date.getDay()));

        const dateStr = formatDateISO(date);
        const dayData = state.heatmap?.[dateStr] || { count: 0, xp: 0 };
        cardsData.push(dayData.count);
        xpData.push(dayData.xp);
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '복습 카드',
                data: cardsData,
                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#334155'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// ===== 포모도로 타이머 =====
function initPomodoro() {
    const modal = document.getElementById('pomodoroModal');
    const openBtn = document.getElementById('pomodoroBtn');
    const closeBtn = document.getElementById('closePomodoroBtn');
    const startBtn = document.getElementById('startTimerBtn');
    const pauseBtn = document.getElementById('pauseTimerBtn');
    const resetBtn = document.getElementById('resetTimerBtn');

    openBtn.addEventListener('click', () => modal.classList.add('active'));
    closeBtn.addEventListener('click', () => modal.classList.remove('active'));

    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });

    startBtn.addEventListener('click', startTimer);
    pauseBtn.addEventListener('click', pauseTimer);
    resetBtn.addEventListener('click', resetTimer);

    updateTimerDisplay();
}

function startTimer() {
    if (state.timer.isRunning) return;

    state.timer.isRunning = true;
    state.timer.isPaused = false;

    document.getElementById('startTimerBtn').style.display = 'none';
    document.getElementById('pauseTimerBtn').style.display = 'block';

    state.timer.interval = setInterval(() => {
        if (state.timer.isPaused) return;

        state.timer.timeLeft--;

        if (state.timer.timeLeft <= 0) {
            handleTimerComplete();
        }

        updateTimerDisplay();
    }, 1000);
}

function pauseTimer() {
    state.timer.isPaused = !state.timer.isPaused;
    document.getElementById('pauseTimerBtn').textContent =
        state.timer.isPaused ? '계속' : '일시정지';
}

function resetTimer() {
    clearInterval(state.timer.interval);
    state.timer.isRunning = false;
    state.timer.isPaused = false;
    state.timer.isBreak = false;
    state.timer.timeLeft = state.timer.workDuration;

    document.getElementById('startTimerBtn').style.display = 'block';
    document.getElementById('pauseTimerBtn').style.display = 'none';
    document.getElementById('timerLabel').textContent = '집중 시간';

    updateTimerDisplay();
}

function handleTimerComplete() {
    clearInterval(state.timer.interval);
    state.timer.isRunning = false;

    // 알림
    if (Notification.permission === 'granted') {
        new Notification(state.timer.isBreak ? '휴식 끝!' : '집중 시간 완료!', {
            body: state.timer.isBreak ? '다음 세션을 시작하세요.' : '휴식을 취하세요!',
            icon: '🍅'
        });
    }

    // 상태 전환
    if (!state.timer.isBreak) {
        state.timer.sessionCount++;
        document.getElementById('sessionCount').textContent = state.timer.sessionCount;

        // 긴 휴식 vs 짧은 휴식
        if (state.timer.sessionCount % 4 === 0) {
            state.timer.timeLeft = state.timer.longBreakDuration;
            document.getElementById('timerLabel').textContent = '긴 휴식';
        } else {
            state.timer.timeLeft = state.timer.breakDuration;
            document.getElementById('timerLabel').textContent = '짧은 휴식';
        }
        state.timer.isBreak = true;
    } else {
        state.timer.timeLeft = state.timer.workDuration;
        document.getElementById('timerLabel').textContent = '집중 시간';
        state.timer.isBreak = false;
    }

    document.getElementById('startTimerBtn').style.display = 'block';
    document.getElementById('pauseTimerBtn').style.display = 'none';

    updateTimerDisplay();
}

function updateTimerDisplay() {
    const minutes = Math.floor(state.timer.timeLeft / 60);
    const seconds = state.timer.timeLeft % 60;
    document.getElementById('timerDisplay').textContent =
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// ===== 툴팁 =====
function initTooltip() {
    // 알림 권한 요청
    if (Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

function showHeatmapTooltip(e) {
    const cell = e.target;
    const tooltip = document.getElementById('heatmapTooltip');

    const date = cell.dataset.date;
    const count = parseInt(cell.dataset.count) || 0;
    const xp = parseInt(cell.dataset.xp) || 0;
    const minutes = parseInt(cell.dataset.minutes) || 0;

    tooltip.innerHTML = `
        <strong>${formatDateKorean(date)}</strong><br>
        복습: ${count}장<br>
        XP: ${xp}<br>
        시간: ${minutes}분
    `;

    const rect = cell.getBoundingClientRect();
    tooltip.style.left = `${rect.left + rect.width / 2}px`;
    tooltip.style.top = `${rect.top - 10}px`;
    tooltip.style.transform = 'translate(-50%, -100%)';
    tooltip.classList.add('visible');
}

function hideHeatmapTooltip() {
    document.getElementById('heatmapTooltip').classList.remove('visible');
}

// ===== 유틸리티 함수 =====
function formatDate(date) {
    const options = { year: 'numeric', month: 'long', day: 'numeric', weekday: 'long' };
    return date.toLocaleDateString('ko-KR', options);
}

function formatDateISO(date) {
    return date.toISOString().split('T')[0];
}

function formatDateKorean(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' });
}

function getMonthName(month) {
    const months = ['1월', '2월', '3월', '4월', '5월', '6월',
                    '7월', '8월', '9월', '10월', '11월', '12월'];
    return months[month];
}

function getDayName(day) {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    return days[day];
}
