import asyncio
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from playwright.async_api import async_playwright, Page
import hashlib
from datetime import datetime
from urllib.parse import urljoin, urlparse
import html
import re
from playwright.async_api import Page

@dataclass
class ScrapedContent:
    """Структура для хранения контента"""
    url: str
    title: str
    level: int
    parent_path: str
    text_content: str
    interactive_content: List[Dict[str, str]]
    links: List[str]
    nested_pages: List[Dict] = None
    timestamp: str = ""
    content_hash: str = ""

class HierarchicalScraper:
    def __init__(self, output_file: str = "abb_bank_hierarchical_data.json", headless: bool = False):
        self.scraped_data = {}
        # Глобальный реестр посещенных URL (нормализованных)
        self.visited_urls: Set[str] = set() 
        self.output_file = output_file
        self.headless = headless
        self.read_only_links = set()
        self.read_nested_links_too = set()

    def normalize_url(self, url: str) -> str:
        """Приводит URL к единому виду, сохраняя важные параметры пагинации"""
        if not url: return ""
        parsed = urlparse(url)
        # Очищаем путь от лишних слэшей
        path = parsed.path.strip().rstrip('/')
        
        # Сохраняем только параметр page, остальное (utm_source и т.д.) отбрасываем
        from urllib.parse import parse_qs, urlencode
        query_params = parse_qs(parsed.query)
        important_params = {}
        if 'page' in query_params:
            important_params['page'] = query_params['page'][0]
        
        new_query = urlencode(important_params)
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if new_query:
            normalized += f"?{new_query}"
            
        return normalized.lower()
        
    def save_current_state(self):
        """Сохраняет текущее состояние данных в файл"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Данные сохранены в {self.output_file}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении: {e}")
    
    def get_content_hash(self, content: str) -> str:
        """Создает хеш контента для индексации"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_internal_link(self, url: str, base_domain: str = "abb-bank.az") -> bool:
        """Проверяет, является ли ссылка внутренней"""
        try:
            parsed = urlparse(url)
            return base_domain in parsed.netloc
        except:
            return False
        
    def is_related_link(self, child_url: str, parent_url: str) -> bool:
        try:
            p_parsed = urlparse(parent_url)
            c_parsed = urlparse(child_url)
            
            parent_path = p_parsed.path.rstrip('/')
            child_path = c_parsed.path.rstrip('/')
            
            # 1. Если это пагинация той же страницы (например, ?page=2)
            if child_path == parent_path and 'page=' in c_parsed.query:
                return True
                
            # 2. Если это вложенная новость или тендер
            if child_path.startswith(parent_path + '/'):
                return True
                
            # 3. Специальные правила для ABB (если в пути есть общие ключи)
            keywords = ['xeberler', 'satinalmalar', 'musabiqelerin-elani']
            if any(kw in parent_path for kw in keywords) and any(kw in child_path for kw in keywords):
                return True
                    
            return False
        except:
            return False
        
    async def process_virtual_scroll_list(self, page: Page, container_selector: str) -> List[Dict]:
        """Специальная обработка для виртуальных списков (как на странице филиалов)"""
        print(f"        🔄 Запуск глубокой прокрутки виртуального списка...")
        
        results = {} # Используем словарь для уникальности по имени филиала
        container = await page.query_selector(container_selector)
        if not container:
            return []

        # Получаем общую высоту контента внутри
        total_height = await page.evaluate('(el) => el.scrollHeight', container)
        viewport_height = await page.evaluate('(el) => el.clientHeight', container)
        
        current_scroll = 0
        step = viewport_height - 100 # Листаем чуть меньше чем на один экран для перекрытия

        while current_scroll < total_height:
            # 1. Скроллим
            await page.evaluate(f'(el) => el.scrollTop = {current_scroll}', container)
            await asyncio.sleep(0.7) # Ждем рендера новых элементов

            # 2. Собираем все видимые в данный момент кнопки
            buttons = await container.query_selector_all('button[type="button"]')
            for btn in buttons:
                title_elem = await btn.query_selector('p.typography-body-hero-regular')
                addr_elem = await btn.query_selector('p.typography-body-compact-regular')
                
                if title_elem:
                    name = (await title_elem.text_content()).strip()
                    address = (await addr_elem.text_content()).strip() if addr_elem else ""
                    
                    if name and name not in results:
                        print(f"          📍 Нашел филиал: {name}")
                        
                        # 3. Кликаем, чтобы получить детали (если нужно)
                        try:
                            await btn.click()
                            await asyncio.sleep(0.8)
                            
                            # Извлекаем данные из открывшейся боковой панели или модалки
                            details = ""
                            # Селектор для деталей (обычно это правая колонка или модалка)
                            detail_panel = await page.query_selector('aside, [class*="sidebar"], [role="dialog"]')
                            if detail_panel:
                                details = await detail_panel.text_content()
                                details = ' '.join(details.split())
                            
                            results[name] = {
                                "title": name,
                                "address": address,
                                "content": details
                            }
                            
                            # Если открылась модалка, закрываем её для следующего шага
                            await page.keyboard.press('Escape')
                        except:
                            results[name] = {"title": name, "address": address, "content": ""}

            current_scroll += step
            # Обновляем общую высоту (на случай динамической подгрузки)
            total_height = await page.evaluate('(el) => el.scrollHeight', container)

        return list(results.values())
    
    async def process_service_network(self, page: Page, url: str) -> List[Dict]:
        """Ультра-надежный сбор Virtual Scroll для ABB Bank"""
        container_sel = '.overflow-y-scroll'
        try:
            await page.wait_for_selector(container_sel, timeout=15000)
            container = await page.query_selector(container_sel)
        except:
            return []
        
        results = {}
        # Получаем размеры
        scroll_height = await page.evaluate('(el) => el.scrollHeight', container)
        
        print(f"        📏 Общая высота: {scroll_height}px. Начинаю сканирование...")

        current_pos = 0
        # Шаг всего 300 пикселей — это примерно 2-3 филиала. 
        # Это гарантирует, что мы не перепрыгнем ни одного.
        step = 300 
        
        # Переменные для контроля прогресса
        stagnant_steps = 0
        last_count = 0

        while current_pos <= (scroll_height + 500):
            # 1. Скроллим
            await page.evaluate('(args) => args.el.scrollTop = args.pos', {'el': container, 'pos': current_pos})
            
            # 2. Ждем чуть дольше (0.7 сек), чтобы React успел перерисовать элементы
            await asyncio.sleep(0.7) 

            # 3. Собираем данные
            buttons = await container.query_selector_all('button[type="button"]')
            for btn in buttons:
                try:
                    title_elem = await btn.query_selector('p.typography-body-hero-regular')
                    if title_elem:
                        name = (await title_elem.text_content()).strip()
                        if name and name not in results:
                            addr_elem = await btn.query_selector('p.typography-body-compact-regular')
                            address = (await addr_elem.text_content()).strip() if addr_elem else ""
                            
                            # Сохраняем (клик пока уберем для скорости, проверим сбор списка)
                            results[name] = {
                                "name": name,
                                "address": address,
                                "details": "Найден в общем списке"
                            }
                            if len(results) % 10 == 0:
                                print(f"          📍 Найдено: {len(results)}...")
                except:
                    continue

            # 4. Проверка на завершение
            if len(results) == last_count:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
                last_count = len(results)

            # Если мы проехали 10 шагов и не нашли ничего нового В САМОМ КОНЦЕ - выходим
            if stagnant_steps > 10 and current_pos > (scroll_height * 0.9):
                break

            current_pos += step
            # Обновляем высоту (на случай динамического расширения)
            scroll_height = await page.evaluate('(el) => el.scrollHeight', container)

        print(f"        ✅ Сбор завершен! Всего: {len(results)} уникальных объектов.")
        return list(results.values())
    
    async def wait_for_page_load(self, page: Page):
        """Ожидание полной загрузки страницы"""
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
            await asyncio.sleep(1)
        except:
            await asyncio.sleep(2)
    
    async def extract_main_content(self, page: Page) -> str:
        """Извлечение основного текстового контента"""
        try:
            main_selectors = ['main', '[role="main"]', 'article', '.content', '.main-content', 'body']
            
            for selector in main_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        if text and len(text.strip()) > 100:
                            cleaned = ' '.join(text.split())
                            print(f"        ✅ Извлечено текста: {len(cleaned)} символов")
                            return cleaned
                except:
                    continue
            
            return ""
        except Exception as e:
            print(f"        ❌ Ошибка при извлечении контента: {e}")
            return ""
    
    async def extract_faq(self, page: Page) -> List[Dict[str, str]]:
        """Извлечение FAQ секций"""
        faq_items = []
        
        faq_selectors = ['.faq-item', '[class*="faq"]', '.accordion-item', 
                        '[class*="accordion"]', '.question-item', '.qa-item']
        
        for selector in faq_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for elem in elements:
                    try:
                        question_selectors = ['.question', '.faq-question', '[class*="question"]',
                                            '.accordion-header', '.accordion-title', 'summary',
                                            'h3', 'h4', '.title', 'button']
                        answer_selectors = ['.answer', '.faq-answer', '[class*="answer"]',
                                          '.accordion-body', '.accordion-content',
                                          '.content', '.description', 'p']
                        
                        question = None
                        answer = None
                        
                        for q_sel in question_selectors:
                            q_elem = await elem.query_selector(q_sel)
                            if q_elem:
                                question = await q_elem.text_content()
                                if question and len(question.strip()) > 5:
                                    question = ' '.join(question.split())
                                    break
                        
                        if question:
                            try:
                                button = await elem.query_selector('button')
                                if button:
                                    await button.click(timeout=1000)
                                    await asyncio.sleep(0.5)
                            except:
                                pass
                            
                            for a_sel in answer_selectors:
                                a_elem = await elem.query_selector(a_sel)
                                if a_elem:
                                    answer = await a_elem.text_content()
                                    if answer and len(answer.strip()) > 10:
                                        answer = ' '.join(answer.split())
                                        break
                        
                        if question and answer and question != answer:
                            faq_item = {'question': question.strip(), 'answer': answer.strip()}
                            if faq_item not in faq_items:
                                faq_items.append(faq_item)
                    except:
                        continue
            except:
                continue
        
        if faq_items:
            print(f"        ✅ Найдено FAQ элементов: {len(faq_items)}")
        
        return faq_items
    
    async def generate_pagination_links(self, page: Page, current_url: str) -> List[str]:
        """Находит максимальную страницу и генерирует список всех URL пагинации"""
        generated_links = []
        try:
            # Селектор для кнопок пагинации (у ABB это обычно внутри nav)
            pagination_elements = await page.query_selector_all('a[href*="page="]')
            
            max_page = 0
            for elem in pagination_elements:
                href = await elem.get_attribute('href')
                if not href: continue
                
                # Извлекаем число после 'page=' (например /xeberler?page=43)
                try:
                    parts = href.split('page=')
                    if len(parts) > 1:
                        page_num = int(parts[1].split('&')[0])
                        if page_num > max_page:
                            max_page = page_num
                except: continue

            if max_page > 0:
                print(f"        🔢 Обнаружена глубокая пагинация: 0 -> {max_page}")
                base_path = current_url.split('?')[0]
                # Генерируем все ссылки. В ABB новости начинаются с page=0
                for i in range(max_page + 1):
                    generated_links.append(f"{base_path}?page={i}")
            else:
                # Если пагинация не найдена через ссылки, пробуем найти текст "страница из X"
                last_page_text = await page.evaluate("""() => {
                    const nav = document.querySelector('nav');
                    return nav ? nav.innerText : "";
                }""")
                # Тут можно добавить regex поиск цифр, если ссылки скрыты за JS
                    
        except Exception as e:
            print(f"        ⚠️ Ошибка генерации пагинации: {e}")
        
        return generated_links
    
    async def extract_all_links(self, page: Page, current_url: str) -> List[str]:
        """Улучшенный сбор ссылок: меню, пагинация и карточки контента"""
        links = set()
        
        # 1. Сначала генерируем ссылки пагинации (если это хаб новостей)
        if "/xeberler" in current_url:
            pagination = await self.generate_pagination_links(page, current_url)
            for p_link in pagination:
                links.add(p_link)

        try:
            # 2. Собираем физические ссылки с текущей страницы
            selectors = [
                'a[href*="/xeberler/"]', 
                'nav[role="navigation"] a[href*="page="]',
                'main a[href]',
                '.card a[href]',
                'a.group' 
            ]
            
            for selector in selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for elem in elements:
                        href = await elem.get_attribute('href')
                        if href:
                            full_url = urljoin(current_url, href)
                            # Очищаем URL от лишних параметров, кроме page
                            if self.is_internal_link(full_url):
                                links.add(full_url)
                except:
                    continue
        except Exception as e:
            print(f"        ❌ Ошибка при извлечении ссылок: {e}")
        
        return list(links)
        
    async def detect_and_process_scrollable_list(self, page: Page, url: str) -> Optional[List[Dict]]:
        """АВТО-ОПРЕДЕЛЕНИЕ и обработка прокручиваемых списков"""
        print(f"        🔍 Поиск прокручиваемых списков...")
        
        try:
            # Ищем контейнеры с overflow-y-scroll
            scroll_containers = await page.query_selector_all('[class*="overflow-y-scroll"], [class*="overflow-y-auto"]')
            
            if not scroll_containers:
                print(f"        ℹ️  Прокручиваемых списков не найдено")
                return None
            
            print(f"        ✅ Найдено контейнеров с прокруткой: {len(scroll_containers)}")
            
            all_items = []
            
            for container_idx, container in enumerate(scroll_containers):
                try:
                    # Проверяем, есть ли внутри кликабельные кнопки
                    buttons = await container.query_selector_all('button[type="button"]')
                    
                    if len(buttons) < 3:  # Минимум 3 элемента, чтобы считать списком
                        continue
                    
                    print(f"        📋 Контейнер {container_idx + 1}: найдено {len(buttons)} кнопок")
                    
                    # Прокручиваем контейнер до конца для загрузки всех элементов
                    await self.scroll_container_to_load_all(page, container)
                    
                    # Переполучаем кнопки после прокрутки
                    buttons = await container.query_selector_all('button[type="button"]')
                    print(f"        📋 После прокрутки: {len(buttons)} элементов")
                    
                    # Обрабатываем каждую кнопку
                    for idx in range(min(len(buttons), 100)):  # Ограничиваем до 100
                        try:
                            # Переполучаем кнопки (могут стать stale)
                            buttons = await container.query_selector_all('button[type="button"]')
                            
                            if idx >= len(buttons):
                                break
                            
                            button = buttons[idx]
                            
                            # Прокручиваем к кнопке
                            await button.scroll_into_view_if_needed()
                            await asyncio.sleep(0.3)
                            
                            # Получаем текст кнопки
                            button_text = await button.text_content()
                            if not button_text or len(button_text.strip()) < 3:
                                continue
                            
                            button_text = button_text.strip()
                            
                            # Пропускаем служебные кнопки
                            skip_texts = ['close', '×', 'geri', 'bağla', 'daxil ol']
                            if any(skip in button_text.lower() for skip in skip_texts):
                                continue
                            
                            print(f"          [{idx+1}/{len(buttons)}] 🖱️  {button_text[:60]}")
                            
                            # Кликаем
                            await button.click(timeout=3000)
                            await asyncio.sleep(1.5)
                            
                            # Извлекаем контент (может быть модальное окно или изменение URL)
                            current_url = page.url
                            
                            detail_content = ""
                            
                            # Проверяем модальное окно
                            modal_selectors = [
                                '[role="dialog"]',
                                '.modal',
                                '[class*="modal"]',
                                'aside',
                                '[class*="sidebar"]'
                            ]
                            
                            modal_found = False
                            for modal_sel in modal_selectors:
                                try:
                                    modal = await page.wait_for_selector(modal_sel, timeout=2000, state='visible')
                                    if modal:
                                        detail_content = await modal.text_content()
                                        detail_content = ' '.join(detail_content.split())
                                        modal_found = True
                                        print(f"          ✅ Модальное окно: {len(detail_content)} символов")
                                        break
                                except:
                                    continue
                            
                            # Если модальное окно не найдено, возможно изменился URL
                            if not modal_found and page.url != current_url:
                                detail_content = await self.extract_main_content(page)
                                await page.go_back()
                                await asyncio.sleep(1)
                                print(f"          ✅ Отдельная страница: {len(detail_content)} символов")
                            
                            if detail_content:
                                all_items.append({
                                    'title': button_text,
                                    'content': detail_content
                                })
                            
                            # Закрываем модальное окно
                            if modal_found:
                                await page.keyboard.press('Escape')
                                await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            print(f"          ⚠️ Ошибка при обработке элемента {idx}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"        ⚠️ Ошибка при обработке контейнера {container_idx}: {e}")
                    continue
            
            if all_items:
                print(f"        ✅ Всего обработано элементов из списков: {len(all_items)}")
                return all_items
            
            return None
            
        except Exception as e:
            print(f"        ❌ Ошибка при обработке прокручиваемых списков: {e}")
            return None
    
    async def scroll_container_to_load_all(self, page: Page, container):
        """Прокрутить контейнер до конца для загрузки всех элементов"""
        try:
            last_height = 0
            attempts = 0
            max_attempts = 30
            
            while attempts < max_attempts:
                # Получаем текущую высоту прокрутки
                current_height = await page.evaluate("""
                    (container) => container.scrollHeight
                """, container)
                
                # Прокручиваем вниз
                await page.evaluate("""
                    (container) => container.scrollTop = container.scrollHeight
                """, container)
                
                await asyncio.sleep(0.5)
                
                # Если высота не изменилась - достигли конца
                if current_height == last_height:
                    break
                
                last_height = current_height
                attempts += 1
            
            # Возвращаемся в начало
            await page.evaluate("""
                (container) => container.scrollTop = 0
            """, container)
            
            await asyncio.sleep(0.5)
            
            print(f"          🔄 Прокрутка завершена за {attempts} попыток")
            
        except Exception as e:
            print(f"          ⚠️ Ошибка прокрутки: {e}")
    
    async def click_and_extract_interactive_content(self, page: Page) -> Tuple[List[Dict[str, str]], str]:
        interactive_content = []
        accumulated_text = ""
        clicked_keys = set() # Храним уникальные ключи (текст + ID)
        
        # Список того, что 100% не несет инфы или ломает поток
        skip_texts = [
            'fərdi', 'biznes', 'korporativ', 'investorlarla', 'haqqımızda', 
            'az', 'en', 'ru', 'daxil ol', 'qeydiyyat', 'tətbiqi yüklə', 
            'apple store', 'google play', 'app gallery', 'search', 'axtar'
        ]

        async def find_new_buttons():
            new_found = []
            # Ищем везде, кроме хедера и футера
            selectors = [
                'button[aria-controls]', 'button[data-state]', 
                'h3 button', '[role="tab"]', '.faq-question'
            ]
            
            for selector in selectors:
                elements = await page.query_selector_all(selector)
                for el in elements:
                    # Проверка: не находится ли кнопка в хедере или футере
                    is_meta = await el.evaluate("""(node) => {
                        return !!node.closest('header') || !!node.closest('footer');
                    }""")
                    if is_meta or not await el.is_visible():
                        continue

                    t = await el.text_content()
                    txt = t.strip().lower() if t else ""
                    
                    if not txt or any(skip == txt for skip in skip_texts) or len(txt) > 80:
                        continue
                    
                    # Уникальный ключ: текст кнопки + ID цели (если есть)
                    controls = await el.get_attribute("aria-controls") or ""
                    key = f"{txt}_{controls}"
                    
                    if key not in clicked_keys:
                        new_found.append((el, key, txt, controls))
            return new_found

        # Увеличиваем лимит итераций для глубокого FAQ
        for i in range(80):
            available = await find_new_buttons()
            if not available:
                break
            
            btn, key, txt, controls_id = available[0]
            clicked_keys.add(key)
            
            print(f"        🖱️ [{i+1}] Клик: '{txt[:40]}'")
            
            try:
                # Снимок всей страницы ДО
                old_page_text = await page.evaluate("document.body.innerText")
                
                await btn.scroll_into_view_if_needed()
                await btn.click(force=True)
                # Для Radix/FAQ нужно время на рендер новых элементов
                await asyncio.sleep(0.8) 

                # 1. Проверка через aria-controls (самый точный метод для аккордеонов)
                content_piece = ""
                if controls_id:
                    content_piece = await page.evaluate(f"""(id) => {{
                        const el = document.getElementById(id);
                        if (!el) return "";
                        el.removeAttribute('hidden');
                        return el.innerText;
                    }}""", controls_id)

                # 2. Если по ID пусто, проверяем, не изменилась ли страница в целом
                new_page_text = await page.evaluate("document.body.innerText")
                
                if (not content_piece or len(content_piece.strip()) < 10) and new_page_text != old_page_text:
                    # Ищем раскрытый блок рядом с кликнутой кнопкой
                    content_piece = await btn.evaluate("""(node) => {
                        const parent = node.parentElement.parentElement;
                        const region = parent.querySelector('[role="region"], [data-state="open"]');
                        return region ? region.innerText : "";
                    }""")

                if content_piece and len(content_piece.strip()) > 5:
                    clean_text = ' '.join(content_piece.split())
                    # Проверяем на дубликаты именно контента
                    if clean_text[:100] not in accumulated_text:
                        accumulated_text += f"\n\n[{txt.upper()}]: {clean_text}"
                        interactive_content.append({'trigger': txt, 'content': clean_text})
                        print(f"        ✅ Собрано")
                    else:
                        print(f"        🔁 Текст уже есть")
                else:
                    # Если текст не найден, возможно кнопка просто открыла список других кнопок (как FAQ)
                    print(f"        👀 Ок, ищем вложенные элементы...")

            except Exception:
                continue

        return interactive_content, accumulated_text

    async def scrape_nested_pages(self, page: Page, base_url: str, links: List[str], level: int) -> List[Dict]:
        """
        Умный сбор вложенных страниц:
        1. Быстро обходит пагинацию (page=1, 2...), собирая только ссылки.
        2. Использует облегченную загрузку (domcontentloaded) для списков.
        3. Качественно скрапит каждую найденную новость.
        """
        nested_data = []
        indent = '  ' * level
        
        # 1. Сортируем и очищаем ссылки
        pagination_links = sorted(list(set([l for l in links if "page=" in l])))
        # Ссылки на новости обычно имеют формат /xeberler/nazvanie-novosti
        all_content_links = set([l for l in links if "/xeberler/" in l and "page=" not in l])
        
        print(f"{indent}🚀 Начало обработки Хаба: {base_url}")

        # 2. ГЛУБОКИЙ СБОР ССЫЛОК СО ВСЕХ СТРАНИЦ ПАГИНАЦИИ
        if pagination_links:
            print(f"{indent}📑 Найдено страниц пагинации: {len(pagination_links)}. Собираем ссылки...")
            
            for p_link in pagination_links:
                if p_link in self.visited_urls:
                    continue
                
                try:
                    # Используем domcontentloaded, чтобы избежать Timeout из-за тяжелых скриптов
                    page_num = p_link.split('page=')[-1]
                    print(f"{indent}  🔍 Сканируем страницу: {page_num}...", end="\r")
                    
                    await page.goto(p_link, wait_until='domcontentloaded', timeout=30000)
                    
                    # Ждем появления карточек новостей (минимум 2 сек, максимум 10)
                    try:
                        await page.wait_for_selector('a[href*="/xeberler/"]', timeout=10000)
                    except:
                        pass # Если нет селектора, попробуем собрать то, что есть

                    # Скролл для активации Lazy Load
                    await page.evaluate("window.scrollBy(0, 600)")
                    await asyncio.sleep(1) 
                    
                    # Собираем ссылки с текущей страницы
                    current_page_links = await self.extract_all_links(page, p_link)
                    
                    # Фильтруем только новости
                    new_news = [l for l in current_page_links if "/xeberler/" in l and "page=" not in l]
                    
                    before_count = len(all_content_links)
                    all_content_links.update(new_news)
                    
                    # Помечаем страницу списка как посещенную
                    self.visited_urls.add(p_link)
                    
                except Exception as e:
                    print(f"\n{indent}    ⚠️ Пропуск страницы пагинации {p_link}: {str(e)[:50]}")
                    continue

        # 3. ПЕРЕХОД К ПОЛНОМУ СКРАПИНГУ КАЖДОЙ НОВОСТИ
        final_news_list = list(all_content_links)
        total_to_scrape = len(final_news_list)
        print(f"\n{indent}🔗 Итого уникальных новостей для анализа: {total_to_scrape}")

        for idx, link in enumerate(final_news_list):
            if link in self.visited_urls:
                continue
            
            print(f"{indent}📄 [{idx+1}/{total_to_scrape}] Скрапинг контента: {link}")
            
            try:
                # Здесь используем стандартный scrape_page с полной загрузкой
                # Указываем parent_path как NewsArchive для иерархии в JSON
                content = await self.scrape_page(page, link, level + 1, "NewsArchive")
                
                if content:
                    from dataclasses import asdict
                    nested_data.append(asdict(content))
                
                # Сохраняем прогресс каждые 10 новостей, чтобы не потерять данные при сбое
                if (idx + 1) % 10 == 0:
                    self.save_current_state()
                    # Небольшая пауза, чтобы не нагружать сервер
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"{indent}  ❌ Ошибка в новости {link}: {e}")

        print(f"{indent}✅ Хаб обработан. Всего новостей в базе: {len(nested_data)}")
        return nested_data
    
    async def scrape_page(self, page: Page, url: str, level: int, parent_path: str) -> Optional[ScrapedContent]:
        """Скрапинг с накоплением текста из всех состояний страницы"""
        
        norm_url = self.normalize_url(url)
        if norm_url in self.visited_urls and level > 0:
            return None
        
        self.visited_urls.add(norm_url)
        indent = '  ' * level
        print(f"\n{indent}🌐 Обработка страницы: {url}")
        
        try:
            # 1. Загрузка страницы
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await self.wait_for_page_load(page)
            
            # Активация Lazy Load (прокрутка вниз и возврат наверх перед кликами)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            await page.evaluate("window.scrollTo(0, 0)")

            # 2. Сбор базового текста страницы (ДО кликов)
            base_clean_text = await page.evaluate("""() => {
                const clone = document.body.cloneNode(true);
                // Удаляем навигацию и футер из базового текста, чтобы не дублировать мусор
                clone.querySelectorAll('script, style, noscript, svg, header, footer').forEach(n => n.remove());
                return clone.innerText;
            }""")

            # 3. Запуск интерактивного сбора (клики по кнопкам/табам/FAQ)
            interactives, clicked_text = await self.click_and_extract_interactive_content(page)

            # 4. "Силовой" сбор всех Radix-блоков (даже если клик не сработал)
            force_extra_text = await page.evaluate("""() => {
                let results = "";
                document.querySelectorAll('[role="region"], [data-state]').forEach(el => {
                    // Собираем текст только если он не скрыт атрибутом hidden
                    if (el.getAttribute('hidden') === null) {
                        const t = el.innerText.trim();
                        if (t.length > 20) results += "\\n" + t;
                    }
                });
                return results;
            }""")

            # 5. Комнуем финальный текст в единую строку
            # Используем список для сборки, чтобы избежать ошибок конкатенации
            final_parts = [base_clean_text]
            
            if clicked_text:
                final_parts.append("\n\n=== РАСКРЫТЫЙ КОНТЕНТ (КЛИКИ) ===\n" + clicked_text)
            
            if force_extra_text:
                # Добавляем только если этого текста еще нет в clicked_text
                if force_extra_text[:100] not in str(clicked_text):
                    final_parts.append("\n\n=== ДОПОЛНИТЕЛЬНЫЕ БЛОКИ ДАННЫХ ===\n" + force_extra_text)

            full_text_result = "".join(final_parts)

            # 6. Сбор ссылок и рекурсия
            links = await self.extract_all_links(page, url)
            
            nested_pages = []
            service_keywords = ['filiallar', 'shobeler', 'atm', 'terminallar']
            is_hub = url in self.read_nested_links_too or any(kw in url for kw in service_keywords)
            
            if is_hub:
                nested_pages = await self.scrape_nested_pages(page, url, links, level)

            return ScrapedContent(
                url=url,
                title=await page.title(),
                level=level,
                parent_path=parent_path,
                text_content=full_text_result, # Теперь содержит всё: базу + клики + FAQ
                interactive_content=interactives, # Список объектов для структурированного анализа
                links=links,
                nested_pages=nested_pages,
                timestamp=datetime.now().isoformat(),
                content_hash=self.get_content_hash(full_text_result)
            )
            
        except Exception as e:
            print(f"{indent}❌ Критическая ошибка на {url}: {e}")
            return None
        
    # async def scrape_page(self, page: Page, url: str, level: int, parent_path: str) -> Optional[ScrapedContent]:
    #     # 1. NORMALIZATION AND DUPLICATE CHECK
    #     norm_url = url.split('#')[0].rstrip('/') # Simple normalization
    #     if norm_url in self.visited_urls:
    #         if level > 0:
    #             print(f"{'  ' * level} ⏩ Skip (Already processed): {url}")
    #             return None
        
    #     self.visited_urls.add(norm_url)
    #     indent = '  ' * level
    #     print(f"\n{indent}🌐 Scraping [{level}]: {url}")

    #     try:
    #         # 2. PAGE LOADING
    #         await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            
    #         # Trigger Lazy Load
    #         await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
    #         await asyncio.sleep(0.5)
    #         await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    #         await asyncio.sleep(1)

    #         title = await page.title()

    #         # 3. SPECIFIC PAGE PROCESSING (Virtual Scroll)
    #         service_keywords = ['filiallar', 'shobeler', 'atm', 'cash-in-atm', 'terminallar', 'xaricde']
    #         scrollable_items = None
    #         interactive_content = []
    #         main_content = ""

    #         if any(kw in url for kw in service_keywords):
    #             print(f"{indent}⚙️ Virtual Scroll detected. Extracting deep data...")
    #             # scrollable_items = await self.process_service_network(page, url)
    #             scrollable_items = [] # Placeholder for your custom method
    #         else:
    #             # 4. NORMAL CONTENT
    #             print(f"{indent}📝 Extracting text and FAQ...")
    #             # main_content = await self.extract_main_content(page)
    #             # faq = await self.extract_faq(page)
                
    #         # 5. LINKS & PAGINATION
    #         links = await page.eval_on_selector_all("a[href]", "elements => elements.map(e => e.href)")
            
    #         # Logic for Hubs (Pagination)
    #         is_pagination = "page=" in url
    #         is_hub = url in self.read_nested_links_too or is_pagination or any(kw in url for kw in service_keywords)

    #         # 6. RECURSIVE CRAWLING
    #         nested_pages = []
    #         if is_hub and level < 3: # Added safety depth limit
    #             print(f"{indent}🌳 Moving to nested pages...")
    #             for link in links[:5]: # Limit for example purposes
    #                 child_content = await self.scrape_page(page, link, level + 1, title)
    #                 if child_content:
    #                     nested_pages.append(child_content)

    #         # 7. FORMING THE RESULT
    #         text_data = main_content if not any(kw in url for kw in service_keywords) else f"Items Count: {len(scrollable_items or [])}"
            
    #         return ScrapedContent(
    #             url=url,
    #             title=title,
    #             level=level,
    #             parent_path=parent_path,
    #             text_content=text_data,
    #             interactive_content=interactive_content,
    #             links=links,
    #             scrollable_items=scrollable_items,
    #             nested_pages=nested_pages,
    #             timestamp=datetime.now().isoformat(),
    #             content_hash="hash_placeholder"
    #         )

    #     except Exception as e:
    #         print(f"{indent}❌ Error on {url}: {e}")
    #         return None
    
    async def scrape_hierarchy(self, hierarchy: Dict, page: Page, level: int = 0, parent_path: str = ""):
        """
        Рекурсивный скрапинг с сохранением древовидной структуры.
        Результат сохраняется в self.scraped_data таким образом, чтобы 
        каждый узел мог содержать и свой контент, и вложенные узлы.
        """
        from dataclasses import asdict

        for key, value in hierarchy.items():
            # 1. Формируем путь (например: Abb/ferdi/kreditler)
            current_path = f"{parent_path}/{key}" if parent_path else key
            path_parts = current_path.split('/')
            
            # 2. Навигация/Создание структуры в self.scraped_data
            # Мы идем по частям пути, создавая вложенные словари, если их нет
            current_level_dict = self.scraped_data
            for part in path_parts:
                if part not in current_level_dict:
                    current_level_dict[part] = {}
                current_level_dict = current_level_dict[part]

            indent = '  ' * level
            print(f"{indent}📁 Обработка узла: {key} (Path: {current_path})")

            # 3. ОПРЕДЕЛЕНИЕ ТИПА ДАННЫХ
            
            # СЦЕНАРИЙ А: Значение — это просто URL (строка)
            if isinstance(value, str):
                content = await self.scrape_page(page, value, level, parent_path)
                if content:
                    # Распаковываем данные скрапинга прямо в этот узел
                    current_level_dict.update(asdict(content))
                    self.save_current_state()
                    await asyncio.sleep(2)

            # СЦЕНАРИЙ Б: Значение — это словарь (с метаданными или вложенными узлами)
            elif isinstance(value, dict):
                node_url = value.get("url")
                
                # Если у этого узла есть свой URL — скрапим его контент
                if node_url:
                    print(f"{indent}📄 Скрапинг контента родительской ноды: {key}")
                    content = await self.scrape_page(page, node_url, level, parent_path)
                    if content:
                        current_level_dict.update(asdict(content))

                # Проверяем наличие вложенных узлов (ключ 'nodes')
                # Если их нет, проверяем просто вложенные ключи (исключая 'url')
                nodes = value.get("nodes")
                if nodes and isinstance(nodes, dict):
                    await self.scrape_hierarchy(nodes, page, level + 1, current_path)
                else:
                    # Если структуры 'nodes' нет, но есть другие ключи (как вложенные разделы)
                    sub_sections = {k: v for k, v in value.items() if k != "url"}
                    if sub_sections:
                        await self.scrape_hierarchy(sub_sections, page, level + 1, current_path)

            # Сохраняем состояние после обработки каждого узла верхнего уровня
            if level == 0:
                self.save_current_state()
    
    async def run(self, hierarchy: Dict, read_only_links: List[str] = None, 
                  read_nested_links_too: List[str] = None):
        """Запуск скрапера"""
        
        if read_only_links:
            self.read_only_links = set(read_only_links)
        if read_nested_links_too:
            self.read_nested_links_too = set(read_nested_links_too)
        
        print("="*80)
        print("🚀 ЗАПУСК ПОЛНОГО ИЕРАРХИЧЕСКОГО СКРАПЕРА ABB BANK")
        print("="*80)
        print(f"📁 Файл сохранения: {self.output_file}")
        print(f"📖 Read-only ссылок: {len(self.read_only_links)}")
        print(f"🌳 Nested scraping ссылок: {len(self.read_nested_links_too)}")
        print(f"👁️  Режим отображения: {'ВКЛЮЧЕН' if not self.headless else 'ВЫКЛЮЧЕН'}")
        print(f"🔍 АВТО-ОПРЕДЕЛЕНИЕ прокручиваемых списков")
        print("="*80)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                slow_mo=500 if not self.headless else 0
            )
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            try:
                await self.scrape_hierarchy(hierarchy, page)
                
                print("\n" + "="*80)
                print("✅ СКРАПИНГ ЗАВЕРШЕН!")
                print("="*80)
                print(f"📊 Всего обработано страниц: {len(self.visited_urls)}")
                print(f"💾 Финальные данные сохранены в: {self.output_file}")
                print("="*80)
                
            except Exception as e:
                print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
                self.save_current_state()
                
            finally:
                await browser.close()


# Ссылки, которые нужно просто прочитать (без захода внутрь найденных на них ссылок)
read_only_links = [
    "https://abb-bank.az/ferdi",
    "https://abb-bank.az/ferdi/kreditler",
    "https://abb-bank.az/ferdi/kartlar",
    "https://abb-bank.az/ferdi/emanetler",
    "https://abb-bank.az/ferdi/butun-hesablar",
    "https://abb-bank.az/ferdi/kesbek",
    "https://abb-bank.az/ferdi/butun-emeliyyatlar",
    "https://abb-bank.az/biznes/korporativ",
    "https://abb-bank.az/biznes/korporativ/korporativ-kreditler",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1",
    "https://abb-bank.az/biznes/korporativ/kocurmeler-1",
    "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar",
    "https://abb-bank.az/biznes/mikro-biznes",
    "https://abb-bank.az/biznes/mikro-biznes/mikro-biznes-kartlar",
    "https://abb-bank.az/biznes/mikro-biznes/odenis-sistemleri",
    "https://abb-bank.az/haqqimizda",
    "https://prime.abb-bank.az"
]

# Ссылки-хабы: скрапер зайдет на них, соберет список (филиалы или новости) и пойдет внутрь каждой ссылки
read_nested_links_too = [
    # xidmet_sebekesi
    "https://abb-bank.az/filiallar",
    "https://abb-bank.az/shobeler",
    "https://abb-bank.az/xaricde",
    "https://abb-bank.az/atm",
    "https://abb-bank.az/cash-in-atm",
    "https://abb-bank.az/terminallar"
]

HIERARCHY = {
    "Abb": {
        "xidmet-sebekesi": {
            "nodes":
            {"filiallar": "https://abb-bank.az/filiallar",
            "shobeler": "https://abb-bank.az/shobeler",
            "xaricde": "https://abb-bank.az/xaricde",
            "atm": "https://abb-bank.az/atm",
            "cash-in-atm": "https://abb-bank.az/cash-in-atm",
            "terminallar": "https://abb-bank.az/terminallar"}
        }
    }
}


async def main():
    scraper = HierarchicalScraper(
        output_file="abb_bank_hierarchical_data_xidmet_sebekesi.json",
        headless=False
    )
    await scraper.run(
        hierarchy=HIERARCHY,
        read_only_links=read_only_links,
        read_nested_links_too=read_nested_links_too
    )

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except RuntimeError:
        asyncio.run(main())