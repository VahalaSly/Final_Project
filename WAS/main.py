import xml.etree.cElementTree as et
import pandas as pd
from tabulate import tabulate

tree = et.parse('summary.xml')
root = tree.getroot()

Id = []
Name = []
State = []
Depth = []
Type = []
Successors = []
MessageType = []
Message = []
i=0

for node in root.iter('node'):
    node_successors = []
    node_message_type = []
    node_message = []
    root1 = node
    for successor in root1.iter("successor"):
        node_successors.append(successor.attrib['id'])
    for nodeMessage in root1.iter("nodeMessage"):
        node_message_type.append(nodeMessage.attrib['type'])
        root2 = nodeMessage
        for message in root2.iter("message"):
            node_message.append(message.text)
    Name.append(node.attrib['name'])
    Id.append(node.attrib['id'])
    State.append(node.attrib['state'])
    Depth.append(node.attrib['graphDepth'])
    Type.append(node.attrib['type'])
    Successors.append(node_successors)
    MessageType.append(node_message_type)
    Message.append(node_message)


df = pd.DataFrame({'Id': Id, 'Name': Name, 'State': State, 'Depth': Depth,
                   'Type': Type, 'MessageType': MessageType,'Message': Message,
                   'Successors': Successors})
print(tabulate(df, headers='keys', tablefmt='psql', showindex='false'))
df.to_excel('summary.xlsx', index=None)
