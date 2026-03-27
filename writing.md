---
layout: default
is_writing: true
---

## Writing

{% for post in site.posts %}
{{ post.date | date: "%Y-%m-%d" }} — [{{ post.title }}]({{ post.url }})
{% endfor %}

---

Older writing on my [Substack](https://thestateoftheunion.substack.com).