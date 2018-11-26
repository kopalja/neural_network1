import requests



with open('pic1.jpg', 'wb') as handle:
        response = requests.get('http://farm1.static.flickr.com/48/146403300_0128782865.jpg', stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)
