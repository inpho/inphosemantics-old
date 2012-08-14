from couchdb import Server

if __name__ = '__main__':
    
    server = Server()

    db = server.create('inpho')
