#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAV Arbeidsplassen job scraper
Usage:
  python nav_job_scraper.py URL [--out output.json]

Notes:
- Uses robust, text-anchored XPath against h2 headings common on arbeidsplassen.nav.no.
- Falls back to CSS heuristics when needed.
- Outputs structured JSON.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import requests
from lxml import html

HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

# ---------------------------- Utilities ----------------------------

def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s, flags=re.MULTILINE).strip()
    return s or None

def first(items: List[Any]) -> Optional[Any]:
    for it in items:
        if it not in (None, "", [], {}):
            return it
    return None

def split_list(s: Optional[str], seps: str = r"[,\n•/]| og | eller ") -> List[str]:
    if not s:
        return []
    parts = re.split(seps, s, flags=re.IGNORECASE)
    return [clean_text(p) for p in parts if clean_text(p)]

def text_after_heading(doc: html.HtmlElement, heading: str) -> Optional[str]:
    # Find h2 equal to heading, get following text block
    xp = f"//h2[normalize-space()='{heading}']/following::*[self::p or self::div or self::span][1]//text()"
    texts = [clean_text(t) for t in doc.xpath(xp)]
    texts = [t for t in texts if t]
    return clean_text(" ".join(texts)) if texts else None

def list_after_heading(doc: html.HtmlElement, heading: str) -> List[str]:
    xp = f"//h2[normalize-space()='{heading}']/following::ul[1]/li//text()"
    items = [clean_text(t) for t in doc.xpath(xp)]
    items = [t for t in items if t]
    # If there were nested texts per li, the above flattens; regroup by li:
    li_nodes = doc.xpath(f"//h2[normalize-space()='{heading}']/following::ul[1]/li")
    results = []
    for li in li_nodes:
        t = " ".join([clean_text(x) for x in li.xpath(".//text()") if clean_text(x)])
        if t:
            results.append(t)
    return results or items

def extract_mail(doc: html.HtmlElement) -> Optional[str]:
    # mailto link first
    mailto = doc.xpath("(//a[starts-with(@href, 'mailto:')]/@href)[1]")
    if mailto:
        return clean_text(mailto[0].replace("mailto:", ""))
    # Fallback: any visible text with @
    txt = doc.xpath("(//a[contains(., '@')] | //text()[contains(., '@')])[1]")
    if isinstance(txt, list) and txt:
        return clean_text(str(txt[0]).replace("mailto:", ""))
    if isinstance(txt, str):
        return clean_text(txt.replace("mailto:", ""))
    return None

def extract_phone(doc: html.HtmlElement) -> Optional[str]:
    # tel: link first
    tel = doc.xpath("(//a[starts-with(@href, 'tel:')]/@href)[1]")
    if tel:
        return re.sub(r"[^0-9+]", "", tel[0].replace("tel:", "")) or None
    # Fallback: 8+ consecutive digits in text
    texts = " ".join([t for t in doc.xpath("//text()") if t])
    m = re.search(r"(\+?\d[\d \-]{6,}\d)", texts)
    if m:
        return re.sub(r"[^\d+]", "", m.group(1))
    return None

def detect_remote(doc: html.HtmlElement) -> Optional[bool]:
    loc_text = text_after_heading(doc, "Sted") or ""
    if re.search(r"hjemmekontor.*ikke mulig", loc_text, flags=re.I):
        return False
    if re.search(r"hjemmekontor.*mulig", loc_text, flags=re.I):
        return True
    return None

def extract_openings(doc: html.HtmlElement) -> Optional[int]:
    raw = text_after_heading(doc, "Antall stillinger")
    if not raw:
        return None
    m = re.search(r"\d+", raw)
    return int(m.group(0)) if m else None

def headings_text(doc: html.HtmlElement, heading: str) -> Optional[str]:
    # More lenient fallback: find heading text contains
    xp = f"//h2[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ', 'abcdefghijklmnopqrstuvwxyzæøå'), '{heading.lower()}')]/following::*[self::p or self::div or self::span][1]//text()"
    texts = [clean_text(t) for t in doc.xpath(xp)]
    texts = [t for t in texts if t]
    return clean_text(' '.join(texts)) if texts else None

# ---------------------------- Extraction ----------------------------

def extract_job(url: str) -> Dict[str, Any]:
    resp = requests.get(url, headers=HDRS, timeout=30)
    resp.raise_for_status()
    doc = html.fromstring(resp.content)

    # Title
    title = first([
        clean_text(first(doc.xpath("//h1//text()"))),
        clean_text(first(doc.xpath("//h1[contains(@class,'title') or contains(@class,'header')]//text()")))
    ])

    # Employer
    employer = first([
        text_after_heading(doc, "Arbeidsgiver"),
        headings_text(doc, "arbeidsgiver")
    ])

    # Position title
    position_title = first([
        text_after_heading(doc, "Stillingstittel"),
        headings_text(doc, "stillingstittel")
    ])

    # Locations
    loc_raw = first([
        text_after_heading(doc, "Sted"),
        headings_text(doc, "sted")
    ])
    locations = split_list(loc_raw) if loc_raw else []

    remote = detect_remote(doc)

    # Start
    start = first([
        text_after_heading(doc, "Oppstart"),
        headings_text(doc, "oppstart")
    ])

    # Employment type
    employment_type = split_list(first([
        text_after_heading(doc, "Type ansettelse"),
        headings_text(doc, "ansettelse")
    ]))

    # Work schedules
    work_schedules = split_list(first([
        text_after_heading(doc, "Arbeidstid"),
        headings_text(doc, "arbeidstid")
    ]))

    # Languages
    languages = split_list(first([
        text_after_heading(doc, "Arbeidsspråk"),
        headings_text(doc, "arbeidsspråk")
    ]), seps=r"[,\n/]| eller | og ")

    openings = extract_openings(doc)

    # Apply/how
    apply_how = first([
        text_after_heading(doc, "Søk på jobben"),
        headings_text(doc, "søk på jobben")
    ])

    apply_url = first(doc.xpath("//h2[normalize-space()='Søk på jobben']/following::a[1]/@href")) or None

    contact_email = extract_mail(doc)
    contact_phone = extract_phone(doc)

    # Contact name: try heading section first
    contact_name = None
    for xp in [
        "//h2[contains(.,'Kontaktperson')]/following::*[self::strong or self::p or self::div][1]//text()",
        "//h2[contains(.,'Søk på jobben')]/following::*[self::strong or self::p or self::div][1]//text()",
    ]:
        txts = [clean_text(t) for t in doc.xpath(xp)]
        txts = [t for t in txts if t]
        if txts:
            # Heuristic: pick the first token-ish name not containing '@' or digits
            for t in txts:
                if '@' in t or re.search(r"\d", t):
                    continue
                if len(t.split()) <= 5:
                    contact_name = t
                    break
        if contact_name:
            break

    # Lists
    what_we_look_for = list_after_heading(doc, "Hva vi ser etter") or []
    tasks = list_after_heading(doc, "Arbeidsoppgaver") or []

    # Qualifications: either under own heading or second list after tasks
    qualifications = list_after_heading(doc, "Ønskede kvalifikasjoner")
    if not qualifications:
        # Try second UL after Arbeidsoppgaver
        ul2 = doc.xpath("//h2[normalize-space()='Arbeidsoppgaver']/following::ul[2]/li")
        if ul2:
            qualifications = [" ".join([clean_text(x) for x in li.xpath(".//text()") if clean_text(x)]) for li in ul2]
            qualifications = [x for x in qualifications if x]

    benefits = list_after_heading(doc, "Vi tilbyr") or []

    # Company
    company_about = first([
        text_after_heading(doc, "Om bedriften"),
        headings_text(doc, "om bedriften"),
        text_after_heading(doc, "Om arbeidsgiver"),
        headings_text(doc, "om arbeidsgiver")
    ])

    company_sector = first([
        text_after_heading(doc, "Sektor"),
        headings_text(doc, "sektor")
    ])

    # Job ID: try to pull from URL first, then Annonsedata
    m = re.search(r"/stilling/([0-9a-fA-F-]{10,})", url)
    job_id = m.group(1) if m else None
    if not job_id:
        ad_txt = first([
            text_after_heading(doc, "Annonsedata"),
            headings_text(doc, "annonsedata")
        ]) or ""
        m2 = re.search(r"[0-9a-fA-F-]{10,}", ad_txt)
        if m2:
            job_id = m2.group(0)

    data = {
        "source_url": url,
        "job_id": job_id,
        "title": title,
        "position_title": position_title,
        "employer": employer,
        "locations": locations,
        "remote": remote,
        "start": start,
        "employment_type": employment_type,
        "work_schedules": work_schedules,
        "languages": languages,
        "openings": openings,
        "apply": {
            "how": apply_how,
            "apply_url": apply_url,
            "contact_email": contact_email,
            "contact_phone": contact_phone,
            "contact_name": contact_name
        },
        "what_we_look_for": what_we_look_for,
        "tasks": tasks,
        "qualifications": qualifications,
        "benefits": benefits,
        "company": {
            "about": company_about,
            "sector": company_sector
        }
    }

    # Drop empty keys for cleanliness
    def prune(obj):
        if isinstance(obj, dict):
            return {k: prune(v) for k, v in obj.items() if v not in (None, "", [], {})}
        if isinstance(obj, list):
            return [prune(x) for x in obj if x not in (None, "", [], {})]
        return obj

    return prune(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Arbeidsplassen job URL")
    ap.add_argument("--out", help="Write JSON to file")
    args = ap.parse_args()

    data = extract_job(args.url)
    js = json.dumps(data, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
    print(js)

if __name__ == "__main__":
    main()
