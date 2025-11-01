"""Links and domains processor for URL extraction and domain parsing."""

import logging
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)


class LinksProcessor:
    """Extract and analyze URLs and domains from text."""

    def __init__(self) -> None:
        """Initialize links processor."""
        
        # Comprehensive URL regex pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:/[^\s]*)?',
            re.IGNORECASE
        )
        
        # Social media domain patterns
        self.social_domains = {
            'twitter.com', 'x.com', 't.co',
            'facebook.com', 'fb.com', 'fb.me',
            'youtube.com', 'youtu.be',
            'instagram.com',
            'linkedin.com',
            'tiktok.com',
            'reddit.com',
            'telegram.me', 't.me',
            'discord.gg',
            'snapchat.com',
            'pinterest.com',
            'tumblr.com'
        }
        
        # News and media domains
        self.news_domains = {
            'cnn.com', 'bbc.com', 'reuters.com', 'ap.org',
            'nytimes.com', 'washingtonpost.com', 'wsj.com',
            'theguardian.com', 'dailymail.co.uk', 'foxnews.com',
            'msnbc.com', 'abc.go.com', 'cbsnews.com', 'nbcnews.com',
            'bloomberg.com', 'ft.com', 'economist.com',
            'politico.com', 'thehill.com', 'axios.com'
        }
        
        # Government domains
        self.gov_domains = {
            'gov', 'mil', 'edu'
        }

    def extract_links(self, text: str) -> Dict[str, Any]:
        """Extract links and analyze domains from text.
        
        Args:
            text: Text to analyze for URLs
            
        Returns:
            Dictionary containing link analysis results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_links()
        
        # Find all URLs
        raw_urls = self.url_pattern.findall(text)
        
        if not raw_urls:
            return self._empty_links()
        
        # Clean and parse URLs
        links = []
        domains = []
        domain_counts = {}
        
        for url in raw_urls:
            try:
                cleaned_url = self._clean_url(url)
                parsed = self._parse_url(cleaned_url)
                
                if parsed['domain']:
                    links.append({
                        'url': cleaned_url,
                        'domain': parsed['domain'],
                        'subdomain': parsed['subdomain'],
                        'path': parsed['path'],
                        'query': parsed['query'],
                        'fragment': parsed['fragment']
                    })
                    
                    domains.append(parsed['domain'])
                    domain_counts[parsed['domain']] = domain_counts.get(parsed['domain'], 0) + 1
                    
            except Exception as e:
                logger.debug(f"Failed to parse URL '{url}': {e}")
                continue
        
        # Analyze domain categories
        domain_categories = self._categorize_domains(domains)
        
        # Calculate statistics
        stats = {
            'total_links': len(links),
            'unique_domains': len(set(domains)),
            'domain_diversity': len(set(domains)) / len(domains) if domains else 0.0,
            'most_frequent_domain': max(domain_counts, key=domain_counts.get) if domain_counts else None,
            'social_media_links': domain_categories['social'],
            'news_media_links': domain_categories['news'],
            'government_links': domain_categories['government'],
            'other_links': domain_categories['other']
        }
        
        return {
            'links': links,
            'domains': domains,
            'domain_counts': domain_counts,
            'domain_categories': domain_categories,
            'stats': stats
        }

    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL."""
        url = url.strip()
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            elif '.' in url and not url.startswith('//'):
                # Likely a domain without protocol
                url = 'https://' + url
        
        return url

    def _parse_url(self, url: str) -> Dict[str, str]:
        """Parse URL into components."""
        try:
            parsed = urlparse(url)
            
            # Extract domain and strip www prefix
            domain = parsed.netloc.lower()
            subdomain = ''
            
            if domain.startswith('www.'):
                domain = domain[4:]
                subdomain = 'www'
            elif '.' in domain:
                parts = domain.split('.')
                if len(parts) > 2:
                    subdomain = parts[0]
                    domain = '.'.join(parts[1:])
            
            return {
                'domain': domain,
                'subdomain': subdomain,
                'path': parsed.path,
                'query': parsed.query,
                'fragment': parsed.fragment
            }
            
        except Exception as e:
            logger.debug(f"URL parsing failed for '{url}': {e}")
            return {
                'domain': '',
                'subdomain': '',
                'path': '',
                'query': '',
                'fragment': ''
            }

    def _categorize_domains(self, domains: List[str]) -> Dict[str, int]:
        """Categorize domains by type."""
        categories = {
            'social': 0,
            'news': 0,
            'government': 0,
            'other': 0
        }
        
        for domain in domains:
            domain_lower = domain.lower()
            
            # Check for social media
            if any(social in domain_lower for social in self.social_domains):
                categories['social'] += 1
            # Check for news media
            elif any(news in domain_lower for news in self.news_domains):
                categories['news'] += 1
            # Check for government domains
            elif any(domain_lower.endswith('.' + gov) for gov in self.gov_domains):
                categories['government'] += 1
            else:
                categories['other'] += 1
        
        return categories

    def _empty_links(self) -> Dict[str, Any]:
        """Return empty links dictionary."""
        return {
            'links': [],
            'domains': [],
            'domain_counts': {},
            'domain_categories': {'social': 0, 'news': 0, 'government': 0, 'other': 0},
            'stats': {
                'total_links': 0,
                'unique_domains': 0,
                'domain_diversity': 0.0,
                'most_frequent_domain': None,
                'social_media_links': 0,
                'news_media_links': 0,
                'government_links': 0,
                'other_links': 0
            }
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Process dataframe and add links and domains data.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added links and domains columns
        """
        logger.info(f"Processing {len(df)} messages for links and domains extraction")
        
        all_link_data = []
        
        for i, row in df.iterrows():
            try:
                text = row[text_column]
                link_data = self.extract_links(text)
                all_link_data.append(link_data)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} messages")
                    
            except Exception as e:
                logger.warning(f"Failed to extract links for message {i}: {e}")
                all_link_data.append(self._empty_links())
        
        # Add links data to dataframe
        df = df.copy()
        
        # Add full link data as JSON columns
        df["links"] = [data["links"] for data in all_link_data]
        df["domains"] = [data["domains"] for data in all_link_data]
        df["domain_counts"] = [data["domain_counts"] for data in all_link_data]
        
        # Add summary statistics as individual columns
        df["total_links"] = [data["stats"]["total_links"] for data in all_link_data]
        df["unique_domains"] = [data["stats"]["unique_domains"] for data in all_link_data]
        df["domain_diversity"] = [data["stats"]["domain_diversity"] for data in all_link_data]
        df["social_media_links"] = [data["stats"]["social_media_links"] for data in all_link_data]
        df["news_media_links"] = [data["stats"]["news_media_links"] for data in all_link_data]
        
        # Log summary statistics
        total_messages_with_links = sum(1 for data in all_link_data if data["stats"]["total_links"] > 0)
        total_links_found = sum(data["stats"]["total_links"] for data in all_link_data)
        total_unique_domains = len(set(
            domain for data in all_link_data for domain in data["domains"]
        ))

        logger.info(f"Links extraction completed:")
        if len(df) > 0:
            logger.info(f"  - Messages with links: {total_messages_with_links}/{len(df)} ({total_messages_with_links/len(df)*100:.1f}%)")
        logger.info(f"  - Total links found: {total_links_found}")
        logger.info(f"  - Unique domains: {total_unique_domains}")
        
        return df

    def export_domain_analysis(self, df: pd.DataFrame, output_path: str) -> None:
        """Export detailed domain analysis to JSON file.
        
        Args:
            df: Processed DataFrame with links data
            output_path: Path to save domain analysis JSON
        """
        import json
        from collections import defaultdict
        from datetime import datetime
        
        # Aggregate domain data across all messages
        domain_stats = defaultdict(lambda: {
            'count': 0,
            'messages': [],
            'first_seen': None,
            'last_seen': None
        })
        
        for i, row in df.iterrows():
            if row['total_links'] > 0:
                msg_date = pd.to_datetime(row.get('date', datetime.now())).strftime('%Y-%m-%d')
                
                for domain in row['domains']:
                    domain_stats[domain]['count'] += 1
                    domain_stats[domain]['messages'].append({
                        'msg_id': row.get('msg_id', i),
                        'date': msg_date,
                        'links_count': len([l for l in row['links'] if l['domain'] == domain])
                    })
                    
                    if domain_stats[domain]['first_seen'] is None:
                        domain_stats[domain]['first_seen'] = msg_date
                    domain_stats[domain]['last_seen'] = msg_date
        
        # Sort domains by frequency
        sorted_domains = dict(sorted(domain_stats.items(), key=lambda x: x[1]['count'], reverse=True))
        
        # Create analysis summary
        analysis = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_messages': len(df),
                'messages_with_links': sum(1 for _, row in df.iterrows() if row['total_links'] > 0),
                'total_links': sum(row['total_links'] for _, row in df.iterrows()),
                'unique_domains': len(domain_stats)
            },
            'top_domains': {
                domain: {
                    'count': stats['count'],
                    'frequency': stats['count'] / len(df),
                    'first_seen': stats['first_seen'],
                    'last_seen': stats['last_seen'],
                    'message_count': len(stats['messages'])
                }
                for domain, stats in list(sorted_domains.items())[:50]  # Top 50
            },
            'domain_categories': self._analyze_domain_categories(sorted_domains),
            'temporal_patterns': self._analyze_temporal_patterns(df)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Domain analysis exported to {output_path}")

    def _analyze_domain_categories(self, domain_stats: Dict) -> Dict:
        """Analyze distribution of domain categories."""
        categories = {
            'social_media': [],
            'news_media': [],
            'government': [],
            'other': []
        }
        
        for domain, stats in domain_stats.items():
            domain_lower = domain.lower()
            
            if any(social in domain_lower for social in self.social_domains):
                categories['social_media'].append({
                    'domain': domain,
                    'count': stats['count']
                })
            elif any(news in domain_lower for news in self.news_domains):
                categories['news_media'].append({
                    'domain': domain,
                    'count': stats['count']
                })
            elif any(domain_lower.endswith('.' + gov) for gov in self.gov_domains):
                categories['government'].append({
                    'domain': domain,
                    'count': stats['count']
                })
            else:
                categories['other'].append({
                    'domain': domain,
                    'count': stats['count']
                })
        
        # Sort each category by count
        for category in categories:
            categories[category] = sorted(categories[category], key=lambda x: x['count'], reverse=True)
        
        return categories

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in link sharing."""
        try:
            df_with_links = df[df['total_links'] > 0].copy()

            if len(df_with_links) == 0:
                return {'daily_link_counts': {}, 'weekly_patterns': {}}

            # Ensure date column exists and is datetime
            if 'date' in df_with_links.columns:
                df_with_links['date'] = pd.to_datetime(df_with_links['date'])
            else:
                df_with_links['date'] = pd.Timestamp.now()

            df_with_links['date_str'] = df_with_links['date'].dt.strftime('%Y-%m-%d')
            df_with_links['weekday'] = df_with_links['date'].dt.day_name()
            
            # Daily patterns
            daily_counts = df_with_links.groupby('date_str').agg({
                'total_links': 'sum',
                'unique_domains': 'sum'
            }).to_dict('index')
            
            # Weekly patterns
            weekly_patterns = df_with_links.groupby('weekday').agg({
                'total_links': 'mean',
                'unique_domains': 'mean'
            }).round(2).to_dict('index')
            
            return {
                'daily_link_counts': daily_counts,
                'weekly_patterns': weekly_patterns
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze temporal patterns: {e}")
            return {'daily_link_counts': {}, 'weekly_patterns': {}}