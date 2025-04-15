WITH RECURSIVE descendants_cte AS (
    -- Base case: Select the most recent 1000 stories
    SELECT 
        id, 
        dead, 
        type, 
        by, 
        time, 
        text, 
        parent, 
        kids, 
        url, 
        score, 
        title, 
        descendants
    FROM hacker_news.items
    WHERE type = 'story'
    ORDER BY time DESC
    LIMIT 10

    UNION ALL

    -- Recursive case: Find all descendants of the base case
    SELECT 
        hn.id, 
        hn.dead, 
        hn.type, 
        hn.by, 
        hn.time, 
        hn.text, 
        hn.parent, 
        hn.kids, 
        hn.url, 
        hn.score, 
        hn.title, 
        hn.descendants
    FROM hacker_news.items hn
    INNER JOIN descendants_cte d
    ON hn.parent = d.id
)
SELECT * 
FROM descendants_cte;