import asyncio
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from playwright.async_api import async_playwright, Page
import hashlib
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
import html
import re


@dataclass
class ScrapedContent:
    """Structure for storing scraped content data."""
    url: str
    title: str
    level: int
    parent_path: str
    text_content: str
    interactive_content: List[Dict[str, str]]
    links: List[str]
    nested_pages: Optional[List[Dict]] = None
    timestamp: str = ""
    content_hash: str = ""


class HierarchicalScraper:
    """Hierarchical web scraper with automatic virtual scroll detection."""
    
    def __init__(
        self,
        output_file: str = "abb_bank_hierarchical_data.json",
        headless: bool = False
    ) -> None:
        """
        Initialize hierarchical scraper.
        
        Args:
            output_file: Path to JSON output file
            headless: Run browser in headless mode
        """
        self.scraped_data: Dict = {}
        # Global registry of visited URLs (normalized)
        self.visited_urls: Set[str] = set()
        self.output_file = output_file
        self.headless = headless
        self.read_only_links: Set[str] = set()
        self.read_nested_links_too: Set[str] = set()

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL to unified format, preserving important pagination parameters.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized lowercase URL
        """
        if not url:
            return ""
        
        parsed = urlparse(url)
        # Clean path from extra slashes
        path = parsed.path.strip().rstrip('/')
        
        # Keep only 'page' parameter, discard others (utm_source, etc.)
        query_params = parse_qs(parsed.query)
        important_params = {}
        if 'page' in query_params:
            important_params['page'] = query_params['page'][0]
        
        new_query = urlencode(important_params)
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if new_query:
            normalized += f"?{new_query}"
            
        return normalized.lower()
        
    def save_current_state(self) -> None:
        """Save current scraped data state to file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Data saved to {self.output_file}")
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def get_content_hash(self, content: str) -> str:
        """
        Create content hash for indexing.
        
        Args:
            content: Text content to hash
            
        Returns:
            16-character hash string
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_internal_link(self, url: str, base_domain: str = "abb-bank.az") -> bool:
        """
        Check if link is internal to base domain.
        
        Args:
            url: URL to check
            base_domain: Base domain to compare against
            
        Returns:
            True if link is internal
        """
        try:
            parsed = urlparse(url)
            return base_domain in parsed.netloc
        except Exception:
            return False
        
    def is_related_link(self, child_url: str, parent_url: str) -> bool:
        """
        Check if child URL is related to parent URL.
        
        Args:
            child_url: Child URL to check
            parent_url: Parent URL to compare against
            
        Returns:
            True if URLs are related
        """
        try:
            p_parsed = urlparse(parent_url)
            c_parsed = urlparse(child_url)
            
            parent_path = p_parsed.path.rstrip('/')
            child_path = c_parsed.path.rstrip('/')
            
            # 1. If it's pagination of the same page (e.g., ?page=2)
            if child_path == parent_path and 'page=' in c_parsed.query:
                return True
                
            # 2. If it's a nested news or tender item
            if child_path.startswith(parent_path + '/'):
                return True
                
            # 3. Special rules for ABB (if path contains common keywords)
            keywords = ['xeberler', 'satinalmalar', 'musabiqelerin-elani']
            if any(kw in parent_path for kw in keywords) and any(kw in child_path for kw in keywords):
                return True
                    
            return False
        except Exception:
            return False
        
    async def process_virtual_scroll_list(
        self,
        page: Page,
        container_selector: str
    ) -> List[Dict]:
        """
        Special processing for virtual scroll lists (like branch pages).
        
        Args:
            page: Playwright page object
            container_selector: CSS selector for scrollable container
            
        Returns:
            List of extracted items with title, address, and content
        """
        print(f"        🔄 Starting deep virtual scroll list processing...")
        
        results = {}  # Use dict for uniqueness by branch name
        container = await page.query_selector(container_selector)
        if not container:
            return []

        # Get total content height
        total_height = await page.evaluate('(el) => el.scrollHeight', container)
        viewport_height = await page.evaluate('(el) => el.clientHeight', container)
        
        current_scroll = 0
        step = viewport_height - 100  # Scroll slightly less than viewport for overlap

        while current_scroll < total_height:
            # 1. Scroll
            await page.evaluate(f'(el) => el.scrollTop = {current_scroll}', container)
            await asyncio.sleep(0.7)  # Wait for new elements to render

            # 2. Collect all currently visible buttons
            buttons = await container.query_selector_all('button[type="button"]')
            for btn in buttons:
                title_elem = await btn.query_selector('p.typography-body-hero-regular')
                addr_elem = await btn.query_selector('p.typography-body-compact-regular')
                
                if title_elem:
                    name = (await title_elem.text_content()).strip()
                    address = (await addr_elem.text_content()).strip() if addr_elem else ""
                    
                    if name and name not in results:
                        print(f"          📍 Found branch: {name}")
                        
                        # 3. Click to get details (if needed)
                        try:
                            await btn.click()
                            await asyncio.sleep(0.8)
                            
                            # Extract data from opened side panel or modal
                            details = ""
                            # Selector for details (usually right column or modal)
                            detail_panel = await page.query_selector(
                                'aside, [class*="sidebar"], [role="dialog"]'
                            )
                            if detail_panel:
                                details = await detail_panel.text_content()
                                details = ' '.join(details.split())
                            
                            results[name] = {
                                "title": name,
                                "address": address,
                                "content": details
                            }
                            
                            # Close modal if opened for next step
                            await page.keyboard.press('Escape')
                        except Exception:
                            results[name] = {
                                "title": name,
                                "address": address,
                                "content": ""
                            }

            current_scroll += step
            # Update total height (in case of dynamic loading)
            total_height = await page.evaluate('(el) => el.scrollHeight', container)

        return list(results.values())
    
    async def process_service_network(self, page: Page, url: str) -> List[Dict]:
        """
        Ultra-reliable virtual scroll collection for ABB Bank.
        
        Args:
            page: Playwright page object
            url: Current page URL
            
        Returns:
            List of collected service network items
        """
        container_sel = '.overflow-y-scroll'
        try:
            await page.wait_for_selector(container_sel, timeout=15000)
            container = await page.query_selector(container_sel)
        except Exception:
            return []
        
        results = {}
        # Get dimensions
        scroll_height = await page.evaluate('(el) => el.scrollHeight', container)
        
        print(f"        📏 Total height: {scroll_height}px. Starting scan...")

        current_pos = 0
        # Step of only 300 pixels - approximately 2-3 branches
        # This ensures we don't skip any items
        step = 300
        
        # Variables for progress control
        stagnant_steps = 0
        last_count = 0

        while current_pos <= (scroll_height + 500):
            # 1. Scroll
            await page.evaluate(
                '(args) => args.el.scrollTop = args.pos',
                {'el': container, 'pos': current_pos}
            )
            
            # 2. Wait a bit longer (0.7 sec) for React to redraw elements
            await asyncio.sleep(0.7)

            # 3. Collect data
            buttons = await container.query_selector_all('button[type="button"]')
            for btn in buttons:
                try:
                    title_elem = await btn.query_selector(
                        'p.typography-body-hero-regular'
                    )
                    if title_elem:
                        name = (await title_elem.text_content()).strip()
                        if name and name not in results:
                            addr_elem = await btn.query_selector(
                                'p.typography-body-compact-regular'
                            )
                            address = (
                                (await addr_elem.text_content()).strip()
                                if addr_elem else ""
                            )
                            
                            # Save (remove click for speed, verify list collection)
                            results[name] = {
                                "name": name,
                                "address": address,
                                "details": "Found in general list"
                            }
                            if len(results) % 10 == 0:
                                print(f"          📍 Found: {len(results)}...")
                except Exception:
                    continue

            # 4. Check for completion
            if len(results) == last_count:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
                last_count = len(results)

            # If we've gone 10 steps without finding anything new AT THE VERY END - exit
            if stagnant_steps > 10 and current_pos > (scroll_height * 0.9):
                break

            current_pos += step
            # Update height (in case of dynamic expansion)
            scroll_height = await page.evaluate('(el) => el.scrollHeight', container)

        print(f"        ✅ Collection complete! Total: {len(results)} unique objects.")
        return list(results.values())
    
    async def wait_for_page_load(self, page: Page) -> None:
        """
        Wait for full page load.
        
        Args:
            page: Playwright page object
        """
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
            await asyncio.sleep(1)
        except Exception:
            await asyncio.sleep(2)
    
    async def extract_main_content(self, page: Page) -> str:
        """
        Extract main text content from page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Cleaned main content text
        """
        try:
            main_selectors = [
                'main', '[role="main"]', 'article',
                '.content', '.main-content', 'body'
            ]
            
            for selector in main_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        if text and len(text.strip()) > 100:
                            cleaned = ' '.join(text.split())
                            print(f"        ✅ Extracted text: {len(cleaned)} characters")
                            return cleaned
                except Exception:
                    continue
            
            return ""
        except Exception as e:
            print(f"        ❌ Content extraction error: {e}")
            return ""
    
    async def extract_faq(self, page: Page) -> List[Dict[str, str]]:
        """
        Extract FAQ sections from page.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of FAQ items with questions and answers
        """
        faq_items = []
        
        faq_selectors = [
            '.faq-item', '[class*="faq"]', '.accordion-item',
            '[class*="accordion"]', '.question-item', '.qa-item'
        ]
        
        for selector in faq_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for elem in elements:
                    try:
                        question_selectors = [
                            '.question', '.faq-question', '[class*="question"]',
                            '.accordion-header', '.accordion-title', 'summary',
                            'h3', 'h4', '.title', 'button'
                        ]
                        answer_selectors = [
                            '.answer', '.faq-answer', '[class*="answer"]',
                            '.accordion-body', '.accordion-content',
                            '.content', '.description', 'p'
                        ]
                        
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
                            except Exception:
                                pass
                            
                            for a_sel in answer_selectors:
                                a_elem = await elem.query_selector(a_sel)
                                if a_elem:
                                    answer = await a_elem.text_content()
                                    if answer and len(answer.strip()) > 10:
                                        answer = ' '.join(answer.split())
                                        break
                        
                        if question and answer and question != answer:
                            faq_item = {
                                'question': question.strip(),
                                'answer': answer.strip()
                            }
                            if faq_item not in faq_items:
                                faq_items.append(faq_item)
                    except Exception:
                        continue
            except Exception:
                continue
        
        if faq_items:
            print(f"        ✅ Found FAQ elements: {len(faq_items)}")
        
        return faq_items
    
    async def generate_pagination_links(
        self,
        page: Page,
        current_url: str
    ) -> List[str]:
        """
        Find maximum page number and generate list of all pagination URLs.
        
        Args:
            page: Playwright page object
            current_url: Current page URL
            
        Returns:
            List of generated pagination URLs
        """
        generated_links = []
        try:
            # Selector for pagination buttons (at ABB usually inside nav)
            pagination_elements = await page.query_selector_all('a[href*="page="]')
            
            max_page = 0
            for elem in pagination_elements:
                href = await elem.get_attribute('href')
                if not href:
                    continue
                
                # Extract number after 'page=' (e.g. /xeberler?page=43)
                try:
                    parts = href.split('page=')
                    if len(parts) > 1:
                        page_num = int(parts[1].split('&')[0])
                        if page_num > max_page:
                            max_page = page_num
                except Exception:
                    continue

            if max_page > 0:
                print(f"        🔢 Deep pagination detected: 0 -> {max_page}")
                base_path = current_url.split('?')[0]
                # Generate all links. At ABB news starts from page=0
                for i in range(max_page + 1):
                    generated_links.append(f"{base_path}?page={i}")
            else:
                # If pagination not found via links, try to find "page X of Y" text
                last_page_text = await page.evaluate("""() => {
                    const nav = document.querySelector('nav');
                    return nav ? nav.innerText : "";
                }""")
                # Here you can add regex search for numbers if links are hidden behind JS
                    
        except Exception as e:
            print(f"        ⚠️ Pagination generation error: {e}")
        
        return generated_links
    
    async def extract_all_links(self, page: Page, current_url: str) -> List[str]:
        """
        Enhanced link collection: menu, pagination and content cards.
        
        Args:
            page: Playwright page object
            current_url: Current page URL
            
        Returns:
            List of extracted links
        """
        links = set()
        
        # 1. First generate pagination links (if this is a news hub)
        if "/xeberler" in current_url:
            pagination = await self.generate_pagination_links(page, current_url)
            for p_link in pagination:
                links.add(p_link)

        try:
            # 2. Collect physical links from current page
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
                            # Clean URL from extra parameters except 'page'
                            if self.is_internal_link(full_url):
                                links.add(full_url)
                except Exception:
                    continue
        except Exception as e:
            print(f"        ❌ Link extraction error: {e}")
        
        return list(links)
        
    async def detect_and_process_scrollable_list(
        self,
        page: Page,
        url: str
    ) -> Optional[List[Dict]]:
        """
        AUTO-DETECT and process scrollable lists.
        
        Args:
            page: Playwright page object
            url: Current page URL
            
        Returns:
            List of processed items or None if no scrollable lists found
        """
        print(f"        🔍 Searching for scrollable lists...")
        
        try:
            # Look for containers with overflow-y-scroll
            scroll_containers = await page.query_selector_all(
                '[class*="overflow-y-scroll"], [class*="overflow-y-auto"]'
            )
            
            if not scroll_containers:
                print(f"        ℹ️  No scrollable lists found")
                return None
            
            print(f"        ✅ Found scroll containers: {len(scroll_containers)}")
            
            all_items = []
            
            for container_idx, container in enumerate(scroll_containers):
                try:
                    # Check if there are clickable buttons inside
                    buttons = await container.query_selector_all('button[type="button"]')
                    
                    if len(buttons) < 3:  # Minimum 3 elements to consider it a list
                        continue
                    
                    print(
                        f"        📋 Container {container_idx + 1}: "
                        f"found {len(buttons)} buttons"
                    )
                    
                    # Scroll container to the end to load all elements
                    await self.scroll_container_to_load_all(page, container)
                    
                    # Re-get buttons after scrolling
                    buttons = await container.query_selector_all('button[type="button"]')
                    print(f"        📋 After scrolling: {len(buttons)} elements")
                    
                    # Process each button
                    for idx in range(min(len(buttons), 100)):  # Limit to 100
                        try:
                            # Re-get buttons (may become stale)
                            buttons = await container.query_selector_all(
                                'button[type="button"]'
                            )
                            
                            if idx >= len(buttons):
                                break
                            
                            button = buttons[idx]
                            
                            # Scroll to button
                            await button.scroll_into_view_if_needed()
                            await asyncio.sleep(0.3)
                            
                            # Get button text
                            button_text = await button.text_content()
                            if not button_text or len(button_text.strip()) < 3:
                                continue
                            
                            button_text = button_text.strip()
                            
                            # Skip service buttons
                            skip_texts = ['close', '×', 'geri', 'bağla', 'daxil ol']
                            if any(skip in button_text.lower() for skip in skip_texts):
                                continue
                            
                            print(
                                f"          [{idx+1}/{len(buttons)}] 🖱️  "
                                f"{button_text[:60]}"
                            )
                            
                            # Click
                            await button.click(timeout=3000)
                            await asyncio.sleep(1.5)
                            
                            # Extract content (may be modal window or URL change)
                            current_url_before = page.url
                            
                            detail_content = ""
                            
                            # Check for modal window
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
                                    modal = await page.wait_for_selector(
                                        modal_sel, timeout=2000, state='visible'
                                    )
                                    if modal:
                                        detail_content = await modal.text_content()
                                        detail_content = ' '.join(detail_content.split())
                                        modal_found = True
                                        print(
                                            f"          ✅ Modal window: "
                                            f"{len(detail_content)} characters"
                                        )
                                        break
                                except Exception:
                                    continue
                            
                            # If modal not found, URL may have changed
                            if not modal_found and page.url != current_url_before:
                                detail_content = await self.extract_main_content(page)
                                await page.go_back()
                                await asyncio.sleep(1)
                                print(
                                    f"          ✅ Separate page: "
                                    f"{len(detail_content)} characters"
                                )
                            
                            if detail_content:
                                all_items.append({
                                    'title': button_text,
                                    'content': detail_content
                                })
                            
                            # Close modal window
                            if modal_found:
                                await page.keyboard.press('Escape')
                                await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            print(f"          ⚠️ Error processing element {idx}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"        ⚠️ Error processing container {container_idx}: {e}")
                    continue
            
            if all_items:
                print(
                    f"        ✅ Total elements processed from lists: "
                    f"{len(all_items)}"
                )
                return all_items
            
            return None
            
        except Exception as e:
            print(f"        ❌ Error processing scrollable lists: {e}")
            return None
    
    async def scroll_container_to_load_all(self, page: Page, container) -> None:
        """
        Scroll container to the end to load all elements.
        
        Args:
            page: Playwright page object
            container: Container element to scroll
        """
        try:
            last_height = 0
            attempts = 0
            max_attempts = 30
            
            while attempts < max_attempts:
                # Get current scroll height
                current_height = await page.evaluate("""
                    (container) => container.scrollHeight
                """, container)
                
                # Scroll down
                await page.evaluate("""
                    (container) => container.scrollTop = container.scrollHeight
                """, container)
                
                await asyncio.sleep(0.5)
                
                # If height hasn't changed - reached the end
                if current_height == last_height:
                    break
                
                last_height = current_height
                attempts += 1
            
            # Return to the beginning
            await page.evaluate("""
                (container) => container.scrollTop = 0
            """, container)
            
            await asyncio.sleep(0.5)
            
            print(f"          🔄 Scrolling completed in {attempts} attempts")
            
        except Exception as e:
            print(f"          ⚠️ Scrolling error: {e}")
    
    async def click_and_extract_interactive_content(
        self,
        page: Page
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        Click interactive elements and extract revealed content.
        
        Args:
            page: Playwright page object
            
        Returns:
            Tuple of (interactive_content_list, accumulated_text)
        """
        interactive_content = []
        accumulated_text = ""
        clicked_keys = set()  # Store unique keys (text + ID)
        
        # List of things that 100% don't carry info or break the flow
        skip_texts = [
            'fərdi', 'biznes', 'korporativ', 'investorlarla', 'haqqımızda',
            'az', 'en', 'ru', 'daxil ol', 'qeydiyyat', 'tətbiqi yüklə',
            'apple store', 'google play', 'app gallery', 'search', 'axtar'
        ]

        async def find_new_buttons():
            new_found = []
            # Search everywhere except header and footer
            selectors = [
                'button[aria-controls]', 'button[data-state]',
                'h3 button', '[role="tab"]', '.faq-question'
            ]
            
            for selector in selectors:
                elements = await page.query_selector_all(selector)
                for el in elements:
                    # Check: is button inside header or footer
                    is_meta = await el.evaluate("""(node) => {
                        return !!node.closest('header') || !!node.closest('footer');
                    }""")
                    if is_meta or not await el.is_visible():
                        continue

                    t = await el.text_content()
                    txt = t.strip().lower() if t else ""
                    
                    if (not txt or any(skip == txt for skip in skip_texts) or
                            len(txt) > 80):
                        continue
                    
                    # Unique key: button text + target ID (if exists)
                    controls = await el.get_attribute("aria-controls") or ""
                    key = f"{txt}_{controls}"
                    
                    if key not in clicked_keys:
                        new_found.append((el, key, txt, controls))
            return new_found

        # Increase iteration limit for deep FAQ
        for i in range(80):
            available = await find_new_buttons()
            if not available:
                break
            
            btn, key, txt, controls_id = available[0]
            clicked_keys.add(key)
            
            print(f"        🖱️ [{i+1}] Click: '{txt[:40]}'")
            
            try:
                # Snapshot of entire page BEFORE
                old_page_text = await page.evaluate("document.body.innerText")
                
                await btn.scroll_into_view_if_needed()
                await btn.click(force=True)
                # For Radix/FAQ need time to render new elements
                await asyncio.sleep(0.8)

                # 1. Check via aria-controls (most precise method for accordions)
                content_piece = ""
                if controls_id:
                    content_piece = await page.evaluate(f"""(id) => {{
                        const el = document.getElementById(id);
                        if (!el) return "";
                        el.removeAttribute('hidden');
                        return el.innerText;
                    }}""", controls_id)

                # 2. If ID is empty, check if page has changed overall
                new_page_text = await page.evaluate("document.body.innerText")
                
                if ((not content_piece or len(content_piece.strip()) < 10) and
                        new_page_text != old_page_text):
                    # Look for expanded block near clicked button
                    content_piece = await btn.evaluate("""(node) => {
                        const parent = node.parentElement.parentElement;
                        const region = parent.querySelector(
                            '[role="region"], [data-state="open"]'
                        );
                        return region ? region.innerText : "";
                    }""")

                if content_piece and len(content_piece.strip()) > 5:
                    clean_text = ' '.join(content_piece.split())
                    # Check for duplicate content
                    if clean_text[:100] not in accumulated_text:
                        accumulated_text += f"\n\n[{txt.upper()}]: {clean_text}"
                        interactive_content.append({
                            'trigger': txt,
                            'content': clean_text
                        })
                        print(f"        ✅ Collected")
                    else:
                        print(f"        🔁 Text already exists")
                else:
                    # If text not found, button may have opened list of other buttons
                    # (like FAQ)
                    print(f"        👀 Ok, looking for nested elements...")

            except Exception:
                continue

        return interactive_content, accumulated_text
    
    async def scrape_nested_pages(
        self,
        page: Page,
        base_url: str,
        links: List[str],
        level: int
    ) -> List[Dict]:
        """
        Smart nested page collection strategy:
        1. Quickly traverses pagination (page=1, 2...), collecting only links
        2. Uses lightweight loading (domcontentloaded) for list pages
        3. Performs quality scraping of each found news item
        
        Args:
            page: Playwright page object
            base_url: Base URL of the hub
            links: List of links to process
            level: Current recursion level
            
        Returns:
            List of nested page data dictionaries
        """
        nested_data: List[Dict] = []
        indent = '  ' * level
           
        # 1. Sort and filter links
        pagination_links = sorted(list(set([l for l in links if "page=" in l])))
        # News links typically have format /xeberler/news-title
        all_content_links: Set[str] = set([
            l for l in links 
            if "/xeberler/" in l and "page=" not in l
        ])
        
        print(f"{indent}🚀 Starting hub processing: {base_url}")

        # 2. DEEP COLLECTION OF LINKS FROM ALL PAGINATION PAGES
        if pagination_links:
            print(f"{indent}📑 Found {len(pagination_links)} pagination pages. Collecting links...")
            
            for p_link in pagination_links:
                if p_link in self.visited_urls:
                    continue
                
                try:
                    # Use domcontentloaded to avoid timeout from heavy scripts
                    page_num = p_link.split('page=')[-1]
                    print(f"{indent}  🔍 Scanning page: {page_num}...", end="\r")
                    
                    await page.goto(p_link, wait_until='domcontentloaded', timeout=30000)
                    
                    # Wait for news cards to appear (min 2 sec, max 10 sec)
                    try:
                        await page.wait_for_selector('a[href*="/xeberler/"]', timeout=10000)
                    except:
                        pass  # If selector not found, try to collect what's available

                    # Scroll to activate lazy loading
                    await page.evaluate("window.scrollBy(0, 600)")
                    await asyncio.sleep(1) 
                    
                    # Collect links from current page
                    current_page_links = await self.extract_all_links(page, p_link)
                    
                    # Filter only news links
                    new_news = [
                        l for l in current_page_links 
                        if "/xeberler/" in l and "page=" not in l
                    ]
                    
                    before_count = len(all_content_links)
                    all_content_links.update(new_news)
                    
                    # Mark list page as visited
                    self.visited_urls.add(p_link)
                    
                except Exception as e:
                    print(f"\n{indent}    ⚠️ Skipping pagination page {p_link}: {str(e)[:50]}")
                    continue

        # 3. PROCEED TO FULL SCRAPING OF EACH NEWS ITEM
        final_news_list = list(all_content_links)
        total_to_scrape = len(final_news_list)
        print(f"\n{indent}🔗 Total unique news items for analysis: {total_to_scrape}")

        for idx, link in enumerate(final_news_list):
            if link in self.visited_urls:
                continue
            
            print(f"{indent}📄 [{idx+1}/{total_to_scrape}] Scraping content: {link}")
            
            try:
                # Here we use standard scrape_page with full loading
                # Set parent_path as NewsArchive for JSON hierarchy
                content = await self.scrape_page(page, link, level + 1, "NewsArchive")
                
                if content:
                    nested_data.append(asdict(content))
                
                # Save progress every 10 news items to prevent data loss on failure
                if (idx + 1) % 10 == 0:
                    self.save_current_state()
                    # Small pause to avoid overloading server
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"{indent}  ❌ Error in news item {link}: {e}")

        print(f"{indent}✅ Hub processed. Total news items in database: {len(nested_data)}")
        return nested_data
    
    async def scrape_page(
        self,
        page: Page,
        url: str,
        level: int,
        parent_path: str
    ) -> Optional['ScrapedContent']:
        """
        Page scraping with text accumulation from all page states.
        
        Strategy:
        1. Load base page content
        2. Execute interactive content collection (clicks on buttons/tabs/FAQ)
        3. Force-collect all Radix blocks
        4. Combine all text into final result
        
        Args:
            page: Playwright page object
            url: URL to scrape
            level: Current recursion level
            parent_path: Parent path in hierarchy
            
        Returns:
            ScrapedContent object or None if already visited
        """
        norm_url = self.normalize_url(url)
        if norm_url in self.visited_urls and level > 0:
            return None
        
        self.visited_urls.add(norm_url)
        indent = '  ' * level
        print(f"\n{indent}🌐 Processing page: {url}")
        
        try:
            # 1. Load page
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await self.wait_for_page_load(page)
            
            # Activate lazy loading (scroll down and return to top before clicks)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            await page.evaluate("window.scrollTo(0, 0)")

            # 2. Collect base page text (BEFORE clicks)
            base_clean_text = await page.evaluate("""() => {
                const clone = document.body.cloneNode(true);
                // Remove navigation and footer from base text to avoid duplicating noise
                clone.querySelectorAll('script, style, noscript, svg, header, footer').forEach(n => n.remove());
                return clone.innerText;
            }""")

            # 3. Run interactive collection (clicks on buttons/tabs/FAQ)
            interactives, clicked_text = await self.click_and_extract_interactive_content(page)

            # 4. "Force" collection of all Radix blocks (even if click didn't work)
            force_extra_text = await page.evaluate("""() => {
                let results = "";
                document.querySelectorAll('[role="region"], [data-state]').forEach(el => {
                    // Collect text only if not hidden by hidden attribute
                    if (el.getAttribute('hidden') === null) {
                        const t = el.innerText.trim();
                        if (t.length > 20) results += "\\n" + t;
                    }
                });
                return results;
            }""")

            # 5. Combine final text into single string
            # Use list for assembly to avoid concatenation errors
            final_parts = [base_clean_text]
            
            if clicked_text:
                final_parts.append("\n\n=== REVEALED CONTENT (CLICKS) ===\n" + clicked_text)
            
            if force_extra_text:
                # Add only if this text is not already in clicked_text
                if force_extra_text[:100] not in str(clicked_text):
                    final_parts.append("\n\n=== ADDITIONAL DATA BLOCKS ===\n" + force_extra_text)

            full_text_result = "".join(final_parts)

            # 6. Collect links and recursion
            links = await self.extract_all_links(page, url)
            
            nested_pages: List[Dict] = []
            service_keywords = ['filiallar', 'shobeler', 'atm', 'terminallar']
            is_hub = url in self.read_nested_links_too or any(kw in url for kw in service_keywords)
            
            if is_hub:
                nested_pages = await self.scrape_nested_pages(page, url, links, level)

            return ScrapedContent(
                url=url,
                title=await page.title(),
                level=level,
                parent_path=parent_path,
                text_content=full_text_result,  # Now contains everything: base + clicks + FAQ
                interactive_content=interactives,  # List of objects for structured analysis
                links=links,
                nested_pages=nested_pages,
                timestamp=datetime.now().isoformat(),
                content_hash=self.get_content_hash(full_text_result)
            )
            
        except Exception as e:
            print(f"{indent}❌ Critical error on {url}: {e}")
            return None
    
    async def scrape_hierarchy(
        self,
        hierarchy: Dict,
        page: Page,
        level: int = 0,
        parent_path: str = ""
    ) -> None:
        """
        Recursive scraping with preservation of tree structure.
        Result is saved in self.scraped_data such that each node
        can contain both its own content and nested nodes.
        
        Args:
            hierarchy: Hierarchical structure to scrape
            page: Playwright page object
            level: Current recursion level
            parent_path: Parent path in hierarchy
        """
        for key, value in hierarchy.items():
            # 1. Form path (e.g., Abb/ferdi/kreditler)
            current_path = f"{parent_path}/{key}" if parent_path else key
            path_parts = current_path.split('/')
            
            # 2. Navigate/create structure in self.scraped_data
            # We go through path parts, creating nested dictionaries if they don't exist
            current_level_dict = self.scraped_data
            for part in path_parts:
                if part not in current_level_dict:
                    current_level_dict[part] = {}
                current_level_dict = current_level_dict[part]

            indent = '  ' * level
            print(f"{indent}📁 Processing node: {key} (Path: {current_path})")

            # 3. DETERMINE DATA TYPE
            
            # SCENARIO A: Value is just a URL (string)
            if isinstance(value, str):
                content = await self.scrape_page(page, value, level, parent_path)
                if content:
                    # Unpack scraping data directly into this node
                    current_level_dict.update(asdict(content))
                    self.save_current_state()
                    await asyncio.sleep(2)

            # SCENARIO B: Value is a dictionary (with metadata or nested nodes)
            elif isinstance(value, dict):
                node_url = value.get("url")
                
                # If this node has its own URL - scrape its content
                if node_url:
                    print(f"{indent}📄 Scraping parent node content: {key}")
                    content = await self.scrape_page(page, node_url, level, parent_path)
                    if content:
                        current_level_dict.update(asdict(content))

                # Check for nested nodes (key 'nodes')
                # If not present, check for nested keys (excluding 'url')
                nodes = value.get("nodes")
                if nodes and isinstance(nodes, dict):
                    await self.scrape_hierarchy(nodes, page, level + 1, current_path)
                else:
                    # If 'nodes' structure doesn't exist, but other keys exist (as nested sections)
                    sub_sections = {k: v for k, v in value.items() if k != "url"}
                    if sub_sections:
                        await self.scrape_hierarchy(sub_sections, page, level + 1, current_path)

            # Save state after processing each top-level node
            if level == 0:
                self.save_current_state()
    
    async def run(
        self,
        hierarchy: Dict,
        read_only_links: Optional[List[str]] = None,
        read_nested_links_too: Optional[List[str]] = None
    ) -> None:
        """
        Run the scraper with provided configuration.
        
        Args:
            hierarchy: Hierarchical structure to scrape
            read_only_links: Links to scrape without following internal links
            read_nested_links_too: Hub links to scrape with nested content
        """
        if read_only_links:
            self.read_only_links = set(read_only_links)
        if read_nested_links_too:
            self.read_nested_links_too = set(read_nested_links_too)
        
        print("=" * 80)
        print("🚀 LAUNCHING FULL HIERARCHICAL ABB BANK SCRAPER")
        print("=" * 80)
        print(f"📁 Output file: {self.output_file}")
        print(f"📖 Read-only links: {len(self.read_only_links)}")
        print(f"🌳 Nested scraping links: {len(self.read_nested_links_too)}")
        print(f"👁️  Display mode: {'ENABLED' if not self.headless else 'DISABLED'}")
        print(f"🔍 AUTO-DETECTION of scrollable lists")
        print("=" * 80)
        
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
                
                print("\n" + "=" * 80)
                print("✅ SCRAPING COMPLETED!")
                print("=" * 80)
                print(f"📊 Total pages processed: {len(self.visited_urls)}")
                print(f"💾 Final data saved to: {self.output_file}")
                print("=" * 80)
                
            except Exception as e:
                print(f"\n❌ CRITICAL ERROR: {e}")
                self.save_current_state()
                
            finally:
                await browser.close()


# ============================================================================
# CONFIGURATION
# ============================================================================


# Links to simply read (without following internal links found on them)
READ_ONLY_LINKS: List[str] = [
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

# Hub links: scraper will visit them, collect list (branches or news) and go inside each link
READ_NESTED_LINKS_TOO: List[str] = [
    # Service network
    "https://abb-bank.az/ferdi/xeberler",
    "https://abb-bank.az/filiallar",
    "https://abb-bank.az/shobeler",
    "https://abb-bank.az/xaricde",
    # ferdi/kreditler
    "https://abb-bank.az/ferdi/kreditler/nagd-kredit",
    "https://abb-bank.az/ferdi/kreditler/avans-kredit-xetti",
    "https://abb-bank.az/ferdi/kreditler/avtomobil-krediti",
    "https://abb-bank.az/ferdi/kreditler/emanetci-nagd-krediti",
    "https://abb-bank.az/ferdi/kreditler/emanetci-kredit-limiti",
    "https://abb-bank.az/ferdi/kreditler/ipoteka-krediti",
    # ferdi/kartlar
    "https://abb-bank.az/ferdi/kartlar/kredit-kartlari",
    "https://abb-bank.az/ferdi/kartlar/debet-kartlari",
    # ferdi/emanetler
    "https://abb-bank.az/ferdi/emanetler/digideposit",
    "https://abb-bank.az/ferdi/emanetler/klassik-emaneti",
    "https://abb-bank.az/ferdi/emanetler/depozit-seyfi",
    "https://abb-bank.az/ferdi/emanetler/emanetli-ipoteka-krediti",
    # ferdi/butun-hesablar
    "https://abb-bank.az/ferdi/butun-hesablar/digihesab-max",
    "https://abb-bank.az/ferdi/butun-hesablar/digihesab",
    "https://abb-bank.az/ferdi/butun-hesablar/cari-hesab",
    "https://abb-bank.az/ferdi/butun-hesablar/dama-dama",
    # ferdi/kesbek
    "https://abb-bank.az/ferdi/kesbek/faydali-kesbek",
    "https://abb-bank.az/ferdi/kesbek/fayda-max",
    "https://abb-bank.az/ferdi/kesbek/faydali-klub",
    # ferdi/kampaniyalar
    "https://abb-bank.az/ferdi/kampaniyalar",
    # ferdi/butun-emeliyyatlar
    "https://abb-bank.az/ferdi/butun-emeliyyatlar/tecili-pul-kocurmeleri",
    "https://abb-bank.az/ferdi/butun-emeliyyatlar/bank-kocurmeleri",
    # ferdi/online-xidmetler
    "https://abb-bank.az/ferdi/randevu",
    "https://abb-bank.az/ferdi/melumat-merkezi",
    "https://abb-bank.az/ferdi/arayis-sifarisi",
    "https://abb-bank.az/ferdi/kredit-odenisi",
    "https://abb-bank.az/ferdi/karta-medaxil",
    "https://abb-bank.az/ferdi/pul-kocurmesi",
    "https://abb-bank.az/ferdi/cash-by-code",
    "https://abb-bank.az/ferdi/iane-et",
    "https://abb-bank.az/ferdi/pin-kod-deyisimi",
    "https://abb-bank.az/ferdi/sigorta",
    "https://abb-bank.az/ferdi/investisiya",
    # biznes/korporativ/korporativ-kreditler
    "https://abb-bank.az/biznes/korporativ/korporativ-kreditler/iri-korporativ-musterilerin-kreditlesdirilmesi",
    "https://abb-bank.az/biznes/korporativ/korporativ-kreditler/ixraca-destek-krediti",
    # biznes/korporativ/odenis-kartlari
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/visa-business-platinum-1",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/mastercard-corporate-travel-expense",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/mastercard-business-1",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/visa-business-1",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/visa-business-gold-1",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/gomruk-karti-1",
    "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/emekhaqqi-kartlari-1",
    # biznes/korporativ/hesablar
    "https://abb-bank.az/biznes/korporativ/hesablar-1",
    # biznes/korporativ/kocurmeler
    "https://abb-bank.az/biznes/korporativ/kocurmeler-1/pul-kocurmeleri-1",
    "https://abb-bank.az/biznes/korporativ/kocurmeler-1/ani-odenis-sistemi",
    # biznes/korporativ/senedli-emeliyyatlar
    "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/qarantiya-1",
    "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/qarantiya-xetti-1",
    "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/akkreditiv-1",
    "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/inkasso-1",
    "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/layihelerin-maliyyelesmesi-1",
    # biznes/kicik-ve-orta-biznes/biznes-kreditleri
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri?kredit-novu=Onlayn+kreditl%C9%99r",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri?kredit-novu=Fiziki+kreditl%C9%99r",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri?kredit-novu=Fond+kreditl%C9%99ri",
    # biznes/kicik-ve-orta-biznes/odenis-kartlari
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/visa-business-platinum",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/mastercard-business",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/visa-business",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/visa-business-gold",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/gomruk-karti",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/emekhaqqi-kartlari",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/bolt-kart",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/sahibkart-visa-paywave",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/sahibkart-mastercard-paypass",
    # biznes/kicik-ve-orta-biznes/kocurmeler
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/kocurmeler",
    # biznes/kicik-ve-orta-biznes/hesablar
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/hesablar",
    # biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/qarantiya",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/qarantiya-xetti",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/akkreditiv",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/inkasso",
    "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/layihelerin-maliyyelesmesi",
    # biznes/mikro-biznes
    "https://abb-bank.az/biznes/mikro-biznes/mikro-biznes-krediti",
    "https://abb-bank.az/biznes/mikro-biznes/bizkart",
    "https://abb-bank.az/biznes/mikro-biznes/e-muhasibim",
    "https://abb-bank.az/biznes/mikro-biznes/odenis-sistemleri/qr-kod",
    "https://abb-bank.az/biznes/mikro-biznes/odenis-sistemleri/mobil-pos",
    "https://abb-bank.az/biznes/mikro-biznes/gundelik-bankciliq",
    # haqqimizda
    "https://abb-bank.az/haqqimizda/rekvizitler",
    "https://abb-bank.az/haqqimizda/missiya-ve-strateji-baxis",
    "https://abb-bank.az/haqqimizda/korporativ-sosial-mesuliyyet",
    "https://abb-bank.az/haqqimizda/muxbir-banklar",
    "https://abb-bank.az/haqqimizda/istirak-payi",
    "https://abb-bank.az/haqqimizda/mukafatlar",
    "https://abb-bank.az/haqqimizda/rehberlik",
    "https://abb-bank.az/haqqimizda/idareetme-ve-komiteler",
    "https://abb-bank.az/haqqimizda/senedler",
    "https://abb-bank.az/haqqimizda/siyasetlerimiz",
    "https://abb-bank.az/korporativ-teqdimat",
    "https://abb-bank.az/haqqimizda/arasdirma",
    "https://abb-bank.az/haqqimizda/hesabatlar",
    "https://abb-bank.az/haqqimizda/teklif-ve-iradlar",
    "https://abb-bank.az/haqqimizda/investisiya-bankciligi",
    "https://abb-bank.az/haqqimizda/brend-merkezi",
    # haqqimizda/satinalmalar
    "https://abb-bank.az/haqqimizda/satinalmalar/musabiqelerin-elani",
    "https://abb-bank.az/haqqimizda/satinalmalar/bildirisler",
    "https://abb-bank.az/haqqimizda/satinalmalar/baglanmis-muqavilelerin-reyestri",
    "https://abb-bank.az/haqqimizda/satinalmalar/satinalma-plani",
    "https://abb-bank.az/haqqimizda/satinalmalar/techizatci-anketi",
    # investorlarla-elaqe
    "https://abb-bank.az/investorlarla-elaqe",
    # abb-premium
    "https://prime.abb-bank.az/aboutus",
    "https://prime.abb-bank.az/products",
    "https://prime.abb-bank.az/services",
    "https://prime.abb-bank.az/specialoffers",
    "https://prime.abb-bank.az/investment",
    "https://prime.abb-bank.az/faq",
    # abb-mobile
    "https://abb-bank.az/abb-mobile",
    # abb-business
    "https://cb.abb-bank.az/digital-platform/az",
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
        "ferdi": {
            "url": "https://abb-bank.az/ferdi",
            "nodes": {
                "kreditler": {
                    "url": "https://abb-bank.az/ferdi/kreditler",
                    "nodes": {
                        "nagd-kredit": "https://abb-bank.az/ferdi/kreditler/nagd-kredit",
                        "avans-kredit-xetti": "https://abb-bank.az/ferdi/kreditler/avans-kredit-xetti",
                        "avtomobil-krediti": "https://abb-bank.az/ferdi/kreditler/avtomobil-krediti",
                        "emanetci-nagd-krediti": "https://abb-bank.az/ferdi/kreditler/emanetci-nagd-krediti",
                        "emanetci-kredit-limiti": "https://abb-bank.az/ferdi/kreditler/emanetci-kredit-limiti",
                        "ipoteka-krediti": "https://abb-bank.az/ferdi/kreditler/ipoteka-krediti"
                    }
                },
                "kartlar": {
                    "url": "https://abb-bank.az/ferdi/kartlar",
                    "nodes": {
                        "kredit-kartlari": "https://abb-bank.az/ferdi/kartlar/kredit-kartlari",
                        "debet-kartlari": "https://abb-bank.az/ferdi/kartlar/debet-kartlari"
                    }
                },
                "emanetler": {
                    "url": "https://abb-bank.az/ferdi/emanetler",
                    "nodes": {
                        "digideposit": "https://abb-bank.az/ferdi/emanetler/digideposit",
                        "klassik-emaneti": "https://abb-bank.az/ferdi/emanetler/klassik-emaneti",
                        "depozit-seyfi": "https://abb-bank.az/ferdi/emanetler/depozit-seyfi",
                        "emanetli-ipoteka-krediti": "https://abb-bank.az/ferdi/emanetler/emanetli-ipoteka-krediti"
                    }
                },
                "butun-hesablar": {
                    "url": "https://abb-bank.az/ferdi/butun-hesablar",
                    "nodes": {
                        "digihesab-max": "https://abb-bank.az/ferdi/butun-hesablar/digihesab-max",
                        "digihesab": "https://abb-bank.az/ferdi/butun-hesablar/digihesab",
                        "cari-hesab": "https://abb-bank.az/ferdi/butun-hesablar/cari-hesab",
                        "dama-dama": "https://abb-bank.az/ferdi/butun-hesablar/dama-dama"
                    }
                },
                "xeberler": "https://abb-bank.az/ferdi/xeberler",
                "kesbek": {
                    "url": "https://abb-bank.az/ferdi/kesbek",
                    "nodes": {
                        "faydali-kesbek": "https://abb-bank.az/ferdi/kesbek/faydali-kesbek",
                        "fayda-max": "https://abb-bank.az/ferdi/kesbek/fayda-max",
                        "faydali-klub": "https://abb-bank.az/ferdi/kesbek/faydali-klub"
                    }
                },
                "kampaniyalar": "https://abb-bank.az/ferdi/kampaniyalar",
                "butun-emeliyyatlar": {
                    "url": "https://abb-bank.az/ferdi/butun-emeliyyatlar",
                    "nodes": {
                        "tecili-pul-kocurmeleri": "https://abb-bank.az/ferdi/butun-emeliyyatlar/tecili-pul-kocurmeleri",
                        "bank-kocurmeleri": "https://abb-bank.az/ferdi/butun-emeliyyatlar/bank-kocurmeleri"
                    }
                },
                "online-xidmetler": {
                    "nodes": {
                        "randevu": "https://abb-bank.az/ferdi/randevu",
                        "melumat-merkezi": "https://abb-bank.az/ferdi/melumat-merkezi",
                        "arayis-sifarisi": "https://abb-bank.az/ferdi/arayis-sifarisi",
                        "kredit-odenisi": "https://abb-bank.az/ferdi/kredit-odenisi",
                        "karta-medaxil": "https://abb-bank.az/ferdi/karta-medaxil",
                        "pul-kocurmesi": "https://abb-bank.az/ferdi/pul-kocurmesi",
                        "cash-by-code": "https://abb-bank.az/ferdi/cash-by-code",
                        "iane-et": "https://abb-bank.az/ferdi/iane-et",
                        "pin-kod-deyisimi": "https://abb-bank.az/ferdi/pin-kod-deyisimi",
                        "sigorta": "https://abb-bank.az/ferdi/sigorta",
                        "investisiya": "https://abb-bank.az/ferdi/investisiya"
                    }
                }
            }
        },
        "biznes": {
            "korporativ": {
                "url": "https://abb-bank.az/biznes/korporativ",
                "nodes": {
                    "korporativ-kreditler": {
                        "url": "https://abb-bank.az/biznes/korporativ/korporativ-kreditler",
                        "nodes": {
                            "iri-korporativ-musterilerin-kreditlesdirilmesi": "https://abb-bank.az/biznes/korporativ/korporativ-kreditler/iri-korporativ-musterilerin-kreditlesdirilmesi",
                            "ixraca-destek-krediti": "https://abb-bank.az/biznes/korporativ/korporativ-kreditler/ixraca-destek-krediti"
                        }
                    },
                    "odenis-kartlari": {
                        "url": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1",
                        "nodes": {
                            "visa-business-platinum": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/visa-business-platinum-1",
                            "mastercard-corporate-travel-expense": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/mastercard-corporate-travel-expense",
                            "mastercard-business": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/mastercard-business-1",
                            "visa-business": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/visa-business-1",
                            "visa-business-gold": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/visa-business-gold-1",
                            "gomruk-karti": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/gomruk-karti-1",
                            "emekhaqqi-kartlari": "https://abb-bank.az/biznes/korporativ/odenis-kartlari-1/emekhaqqi-kartlari-1"
                        }
                    },
                    "hesablar": "https://abb-bank.az/biznes/korporativ/hesablar-1",
                    "kocurmeler": {
                        "url": "https://abb-bank.az/biznes/korporativ/kocurmeler-1",
                        "nodes": {
                            "pul-kocurmeleri": "https://abb-bank.az/biznes/korporativ/kocurmeler-1/pul-kocurmeleri-1",
                            "ani-odenis-sistemi": "https://abb-bank.az/biznes/korporativ/kocurmeler-1/ani-odenis-sistemi"
                        }
                    },
                    "senedli-emeliyyatlar": {
                        "url": "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1",
                        "nodes": {
                            "qarantiya": "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/qarantiya-1",
                            "qarantiya-xetti": "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/qarantiya-xetti-1",
                            "akkreditiv": "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/akkreditiv-1",
                            "inkasso": "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/inkasso-1",
                            "layihelerin-maliyyelesmesi": "https://abb-bank.az/biznes/korporativ/senedli-emeliyyatlar-1/layihelerin-maliyyelesmesi-1"
                        }
                    }
                }
            },
            "kicik-ve-orta-biznes": {
                "url": "https://abb-bank.az/biznes/kicik-ve-orta-biznes",
                "nodes": {
                    "biznes-kreditleri": {
                        "url": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri",
                        "nodes": {
                            "onlayn-kreditler": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri?kredit-novu=Onlayn+kreditl%C9%99r",
                            "fiziki-kreditler": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri?kredit-novu=Fiziki+kreditl%C9%99r",
                            "fond-kreditleri": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/biznes-kreditleri?kredit-novu=Fond+kreditl%C9%99ri"
                        }
                    },
                    "odenis-kartlari": {
                        "url": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari",
                        "nodes": {
                            "visa-business-platinum": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/visa-business-platinum",
                            "mastercard-business": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/mastercard-business",
                            "visa-business": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/visa-business",
                            "visa-business-gold": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/visa-business-gold",
                            "gomruk-karti": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/gomruk-karti",
                            "emekhaqqi-kartlari": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/emekhaqqi-kartlari",
                            "bolt-kart": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/bolt-kart",
                            "sahibkart-visa-paywave": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/sahibkart-visa-paywave",
                            "sahibkart-mastercard-paypass": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/odenis-kartlari/sahibkart-mastercard-paypass"
                        }
                    },
                    "kocurmeler": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/kocurmeler",
                    "hesablar": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/hesablar",
                    "senedli-emeliyyatlar": {
                        "url": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar",
                        "nodes": {
                            "qarantiya": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/qarantiya",
                            "qarantiya-xetti": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/qarantiya-xetti",
                            "akkreditiv": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/akkreditiv",
                            "inkasso": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/inkasso",
                            "layihelerin-maliyyelesmesi": "https://abb-bank.az/biznes/kicik-ve-orta-biznes/senedli-emeliyyatlar/layihelerin-maliyyelesmesi"
                        }
                    }
                }
            },
            "mikro-biznes": {
                "url": "https://abb-bank.az/biznes/mikro-biznes",
                "nodes": {
                    "mikro-biznes-krediti": "https://abb-bank.az/biznes/mikro-biznes/mikro-biznes-krediti",
                    "mikro-biznes-kartlar": {
                        "url": "https://abb-bank.az/biznes/mikro-biznes/mikro-biznes-kartlar",
                        "nodes": {
                            "bizkart": "https://abb-bank.az/biznes/mikro-biznes/bizkart"
                        }
                    },
                    "e-muhasibim": "https://abb-bank.az/biznes/mikro-biznes/e-muhasibim",
                    "odenis-sistemleri": {
                        "url": "https://abb-bank.az/biznes/mikro-biznes/odenis-sistemleri",
                        "nodes": {
                            "qr-kod": "https://abb-bank.az/biznes/mikro-biznes/odenis-sistemleri/qr-kod",
                            "mobil-pos": "https://abb-bank.az/biznes/mikro-biznes/odenis-sistemleri/mobil-pos"
                        }
                    },
                    "gundelik-bankciliq": "https://abb-bank.az/biznes/mikro-biznes/gundelik-bankciliq"
                }
            }
        },
        "haqqimizda": {
            "url": "https://abb-bank.az/haqqimizda",
            "nodes": {
                "rekvizitler": "https://abb-bank.az/haqqimizda/rekvizitler",
                "missiya-ve-strateji-baxis": "https://abb-bank.az/haqqimizda/missiya-ve-strateji-baxis",
                "korporativ-sosial-mesuliyyet": "https://abb-bank.az/haqqimizda/korporativ-sosial-mesuliyyet",
                "muxbir-banklar": "https://abb-bank.az/haqqimizda/muxbir-banklar",
                "istirak-payi": "https://abb-bank.az/haqqimizda/istirak-payi",
                "mukafatlar": "https://abb-bank.az/haqqimizda/mukafatlar",
                "rehberlik": "https://abb-bank.az/haqqimizda/rehberlik",
                "idareetme-ve-komiteler": "https://abb-bank.az/haqqimizda/idareetme-ve-komiteler",
                "senedler": "https://abb-bank.az/haqqimizda/senedler",
                "siyasetlerimiz": "https://abb-bank.az/haqqimizda/siyasetlerimiz",
                "korporativ-teqdimat": "https://abb-bank.az/korporativ-teqdimat",
                "arasdirma": "https://abb-bank.az/haqqimizda/arasdirma",
                "hesabatlar": "https://abb-bank.az/haqqimizda/hesabatlar",
                "teklif-ve-iradlar": "https://abb-bank.az/haqqimizda/teklif-ve-iradlar",
                "investisiya-bankciligi": "https://abb-bank.az/haqqimizda/investisiya-bankciligi",
                "brend-merkezi": "https://abb-bank.az/haqqimizda/brend-merkezi",
                "satinalmalar": {
                    "nodes": {
                        "musabiqelerin-elani": "https://abb-bank.az/haqqimizda/satinalmalar/musabiqelerin-elani",
                        "bildirisler": "https://abb-bank.az/haqqimizda/satinalmalar/bildirisler",
                        "baglanmis-muqavilelerin-reyestri": "https://abb-bank.az/haqqimizda/satinalmalar/baglanmis-muqavilelerin-reyestri",
                        "satinalma-plani": "https://abb-bank.az/haqqimizda/satinalmalar/satinalma-plani",
                        "techizatci-anketi": "https://abb-bank.az/haqqimizda/satinalmalar/techizatci-anketi"
                    }
                }
            }
        },
        "investorlarla-elaqe": "https://abb-bank.az/investorlarla-elaqe",
        "abb-premium": {
            "url": "https://prime.abb-bank.az",
            "nodes": {
                "aboutus": "https://prime.abb-bank.az/aboutus",
                "products": "https://prime.abb-bank.az/products",
                "services": "https://prime.abb-bank.az/services",
                "specialoffers": "https://prime.abb-bank.az/specialoffers",
                "investment": "https://prime.abb-bank.az/investment",
                "faq": "https://prime.abb-bank.az/faq"
            }
        },
        "abb-mobile": "https://abb-bank.az/abb-mobile",
        "abb-business": "https://cb.abb-bank.az/digital-platform/az",
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main() -> None:
    """Main entry point for the scraper."""
    scraper = HierarchicalScraper(
        output_file="abb_bank_hierarchical_data_xidmet_sebekesi.json",
        headless=False
    )
    await scraper.run(
        hierarchy=HIERARCHY,
        read_only_links=READ_ONLY_LINKS,
        read_nested_links_too=READ_NESTED_LINKS_TOO
    )


if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except RuntimeError:
        asyncio.run(main())