import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

import main
from digest.article_cache import ArticleCache
from digest.company_sources import filter_company_articles, is_recent_publication, parse_company_blog, publication_date
from digest.config import Config
from digest.models import Article
from digest.parser import collect_articles, parse_feed

NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


class CompanySourcesTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("digest.company_sources.datetime", FixedDatetime)
        patcher.start()
        self.addCleanup(patcher.stop)

    def html_client(self, documents):
        requested = []
        real_client = httpx.Client

        def respond(request):
            url = str(request.url)
            requested.append(url)
            body = documents.get(url)
            return httpx.Response(200 if body is not None else 404, text=body or "missing")

        patcher = patch("digest.company_sources.httpx.Client", side_effect=lambda **kwargs:
                        real_client(transport=httpx.MockTransport(respond), **kwargs))
        patcher.start()
        self.addCleanup(patcher.stop)
        return requested

    def test_date_formats_timezone_and_week_boundaries(self):
        for value in ("2026-09-11", "2026-09-11T00:00:00Z", "Fri, 11 Sep 2026 00:00:00 GMT",
                      "Sep 11, 2026", "September 11, 2026"):
            with self.subTest(value=value):
                self.assertTrue(is_recent_publication(value, NOW))
        self.assertEqual(publication_date("2026-09-11T03:00:00+03:00"), publication_date("2026-09-11"))
        for age, recent in ((timedelta(days=7), True), (timedelta(days=7, seconds=1), False),
                            (timedelta(seconds=-1), False)):
            self.assertEqual(is_recent_publication((NOW - age).isoformat(), NOW), recent)
        for value in ("", "Sep 11", "yesterday", None):
            self.assertFalse(is_recent_publication(value, NOW))

    def test_rss_uses_publication_date_and_never_recent_update_of_old_article(self):
        rss = '''<rss version="2.0"><channel><title>Models</title>
          <item><title>New model</title><link>https://deepmind.google/blog/new/</link>
            <pubDate>Fri, 11 Sep 2026 00:00:00 GMT</pubDate><description>Useful details</description></item>
          <item><title>Old model</title><link>https://deepmind.google/blog/old/</link>
            <pubDate>Mon, 01 Jun 2026 00:00:00 GMT</pubDate></item>
          <item><title>Undated</title><link>https://deepmind.google/blog/undated/</link></item>
          <item><title>Future</title><link>https://deepmind.google/blog/future/</link>
            <pubDate>Wed, 16 Sep 2026 00:00:00 GMT</pubDate></item>
          </channel></rss>'''
        with patch("digest.parser.httpx.get", return_value=httpx.Response(
            200, text=rss, request=httpx.Request("GET", "https://example.com/rss"),
        )) as get:
            items = parse_feed("https://deepmind.google/blog/rss.xml", "deepmind")
        self.assertEqual([a.title for a in items], ["New model"])
        self.assertEqual(get.call_args.kwargs["timeout"], 20)

        atom = '''<feed xmlns="http://www.w3.org/2005/Atom"><title>Models</title>
          <entry><title>Recently edited old article</title><link href="https://example.com/old"/>
            <published>2026-06-01T00:00:00Z</published><updated>2026-09-14T00:00:00Z</updated></entry>
          <entry><title>Only updated</title><link href="https://example.com/unknown"/>
            <updated>2026-09-14T00:00:00Z</updated></entry></feed>'''
        with patch("digest.parser.httpx.get", return_value=httpx.Response(
            200, text=atom, request=httpx.Request("GET", "https://example.com/rss"),
        )):
            self.assertEqual(parse_feed("https://deepmind.google/blog/rss.xml", "deepmind"), [])

    def test_openai_verifies_year_on_article_and_ignores_navigation_and_body_dates(self):
        base = "https://developers.openai.com/blog/"
        listing = '''<a href="/blog/nav">Navigation</a>
            <a class="resource-item" href="/blog/new"><div>Sep 11</div><p>New preview</p></a>
            <a class="resource-item" href="/blog/old"><div>Sep 11</div></a>
            <a class="resource-item" href="/blog/no-date"><div>Sep 11</div></a>
            <a class="resource-item" href="/blog/august"><div>Aug 10</div></a>'''
        requested = self.html_client({base: listing,
            base + "new": '<main><span class="text-default font-medium">Sep 11, 2026</span><h1>New</h1></main>',
            base + "old": '<main><span class="text-default font-medium">Sep 11, 2025</span><h1>Old</h1></main>',
            base + "no-date": '<main><h1>No date</h1><p>Sep 11, 2026</p></main>',
        })
        items = parse_company_blog(base, "openai", now=NOW)
        self.assertEqual([a.title for a in items], ["New"])
        self.assertEqual(items[0].preview, "New preview")
        self.assertNotIn(base + "nav", requested)
        self.assertNotIn(base + "august", requested)

    def test_anthropic_jsonld_date_published_and_team_exclusion(self):
        base = "https://www.anthropic.com/research"
        def article_page(published, modified):
            return '<main><h1>Research paper</h1></main><script type="application/ld+json">' + json.dumps({
                "@graph": [{"@type": "BlogPosting", "datePublished": published,
                            "dateModified": modified, "description": "Research preview"}],
            }) + '</script>'
        requested = self.html_client({base: '''
            <article><a href="/research/new">New paper</a></article>
            <article><a href="/research/old">Old paper</a></article>
            <article><a href="/research/team/alignment">Alignment team</a></article>
            ''',
            base + "/new": article_page("2026-09-11", "2026-09-14"),
            base + "/old": article_page("2026-06-01", "2026-09-14"),
        })
        items = parse_company_blog(base, "anthropic", now=NOW)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, base + "/new")
        self.assertNotIn(base + "/team/alignment", requested)

    def test_meta_listing_dates_skip_old_cards_and_keep_confirmed_publication_date(self):
        base = "https://ai.meta.com/blog/"
        requested = self.html_client({base: '''
            <div><a class="_amdf" href="/blog/new/">New model</a><div>Sep 11, 2026</div></div>
            <div><a class="_amdf" href="/blog/old/">Old model</a><div>Jun 01, 2026</div></div>
            ''',
            base + "new/": '<main><h1>New model</h1><p>' + 'Practical research details. ' * 10 + '</p></main>',
        })
        items = parse_company_blog(base, "meta", now=NOW)
        self.assertEqual([a.title for a in items], ["New model"])
        self.assertNotIn(base + "old/", requested)

    def test_one_unavailable_article_does_not_discard_other_articles(self):
        base = "https://www.anthropic.com/engineering"
        self.html_client({base: '''<article><a href="/engineering/missing">Missing</a></article>
            <article><a href="/engineering/new">New</a></article>''',
            base + "/new": '<main><h1>New</h1><p class="date">Published Sep 11, 2026</p></main>',
        })
        self.assertEqual(len(parse_company_blog(base, "anthropic", now=NOW)), 1)

    def test_source_failure_does_not_stop_other_sources_or_habr_stats(self):
        recent = Article("New", "https://example.com/new", "deepmind", "", "2026-09-11")
        with patch("digest.parser.FEEDS", {"openai": [("Blog", "bad")], "deepmind": [("Blog", "good")]}), \
                patch("digest.parser.parse_feed", side_effect=[httpx.ConnectError("Unavailable"), [recent]]), \
                patch("digest.parser.fetch_habr_stats") as stats:
            self.assertEqual(collect_articles(fetch_stats=False), [recent])
        stats.assert_not_called()

    def test_week_filter_leaves_existing_sources_unchanged(self):
        habr = Article("Legacy", "https://example.com/old", "habr", "", "")
        old = Article("Old company post", "https://example.com/company", "mistral", "", "2026-06-01")
        self.assertEqual(filter_company_articles([habr, old], NOW), [habr])

    def test_cached_company_article_is_removed_when_it_passes_seven_days(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ArticleCache(root / "articles_cache.json")
            published = NOW - timedelta(days=7, hours=1)
            item = Article("Old now", "https://example.com/post", "openai", "", published.isoformat())
            # Pool was parsed two hours ago, when this article was still in the seven-day window.
            with patch("digest.article_cache.time.time", return_value=(NOW - timedelta(hours=2)).timestamp()):
                cache.save([item])
            with patch.dict("os.environ", {"DIGEST_DATA_DIR": str(root)}), \
                    patch.object(main, "ARTICLES_FILE", root / "articles.json"), \
                    patch("digest.article_cache.time.time", return_value=NOW.timestamp()), \
                    patch.object(main, "collect_articles") as collect, patch.object(main, "get_provider") as provider:
                self.assertEqual(main.generate_digest(Config(), use_article_cache=True), ([], []))
            collect.assert_not_called()
            provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
